# Production Deployment Guide: InsureQuery API

This guide provides the complete, production-grade instructions for deploying the **InsureQuery API** on a single Linux VPS behind **Caddy** (automatic HTTPS) at `api.prince-gupta.me`, with **Docker Compose**, strict security hardening, and zero exposed database ports.

---

## 🏗️ Architecture Overview

```
                      Internet
                         │
          ┌──────────────┴──────────────┐
          │   https://api.prince-gupta.me
          ▼
   ┌──────────────────────────────────────────────┐
   │                  Caddy:2                     │
   │  - Automatic TLS (Let's Encrypt / ZeroSSL)   │
   │  - Ports 80 & 443 exposed to host            │
   │  - /metrics hidden (HTTP 404 response)       │
   │  - Unbuffered SSE stream (flush_interval -1) │
   │  - Max body size 30MB                        │
   └──────────────────────┬───────────────────────┘
                          │ (Internal Docker Network)
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│     insurequery-api       │   │           redis           │
│  - Non-root user: app     │   │  - Port 6379 INTERNAL     │
│  - Internal port: 8000    │   │  - NO host port published │
│  - Fail-closed Bearer auth│◄──┤  - LRU memory eviction    │
│  - CPU-only Torch wheel   │   │  - Embedding cache        │
│  - 1 uvicorn worker       │   └───────────────────────────┘
└─────────────┬─────────────┘
              │
      ┌───────┴───────┐
      ▼               ▼
┌───────────┐   ┌───────────┐
│insure_data│   │ hf_cache  │ (Persistent Named Volumes)
└───────────┘   └───────────┘
```

---

## 📋 Security Hardening Checklist (Implemented)

1. **Fail-Closed Authentication**:
   - `verify_auth` in `main_api.py` checks for `INSUREQUERY_API_KEY`. If unset, it raises `503 Service Unavailable` instead of allowing unauthenticated requests.
   - All tokens are verified using `secrets.compare_digest(token, configured_key)` to eliminate timing attacks.
2. **SSRF Attack Surface Elimination**:
   - URL-fetching routes (`/documents/from-url` and `/hackrx/run`) are completely removed from the production API to prevent Server-Side Request Forgery against cloud metadata endpoints (`169.254.169.254`) or internal services.
3. **No Unauthenticated Port Exposure**:
   - Redis port `6379` and FastAPI port `8000` are **NOT** published on the host network. Docker Compose bypasses host `ufw` rules for published ports, so omitting `ports:` and using `expose: ["8000"]` ensures they are only reachable via Docker's internal bridge network.
4. **CORS Locked**:
   - CORS is restricted to your production domain (`https://prince-gupta.me`) with `allow_credentials=False`.
5. **Non-Root Container Execution**:
   - The Dockerfile provisions an unprivileged `app` user (`UID/GID 1000`) and drops root privileges with `USER app`.
6. **Pre-cached CPU Models & Slim Build**:
   - Dependencies are slimmed (dev tools and Streamlit removed from API build).
   - CPU-only PyTorch (`torch==2.3.0`) is installed from PyTorch's official wheel index.
   - Hugging Face models (`all-MiniLM-L6-v2` and `ms-marco-MiniLM-L-6-v2`) are downloaded into `/home/app/.cache/huggingface` during the Docker image build.
7. **Protected Internal Metrics**:
   - Caddy blocks external access to `GET /metrics` (`respond @metrics 404`), keeping internal latency counters and cache metrics confidential.
8. **Unbuffered SSE Streaming**:
   - Caddy proxies `/query/stream` with `flush_interval -1` to preserve instant token-by-token streaming over SSE.

---

## 🛠️ Step 1: Provision the Server

Use a Linux Virtual Private Server (VPS) with:
- **CPU**: 2+ vCPUs
- **RAM**: Minimum **4 GB** (Sentence-Transformers + Cross-Encoder + ChromaDB in memory require ~2.5 GB peak)
- **Disk**: 40+ GB SSD/NVMe
- **OS**: Ubuntu 24.04 LTS (x86_64)
- **Providers**: Hetzner Cloud (e.g. CPX21), DigitalOcean, Linode, or AWS EC2 (`t3.medium` or `c6i.large`).
- **SSH**: Attach your public SSH key (`~/.ssh/id_ed25519.pub`) upon instance creation.

---

## 🔒 Step 2: Server Hardening & Base Setup

SSH into the server as root:
```bash
ssh root@<SERVER_IP>
```

### 1. Create a Non-Root Admin User
```bash
adduser deployer
usermod -aG sudo deployer

# Copy SSH authorized keys from root
mkdir -p /home/deployer/.ssh
cp /root/.ssh/authorized_keys /home/deployer/.ssh/
chown -R deployer:deployer /home/deployer/.ssh
chmod 700 /home/deployer/.ssh
chmod 600 /home/deployer/.ssh/authorized_keys
```

### 2. Harden SSH Daemon
Edit `/etc/ssh/sshd_config.d/hardening.conf` (or `/etc/ssh/sshd_config`):
```ini
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
X11Forwarding no
MaxAuthTries 4
```
Validate and restart SSH:
```bash
sshd -t && systemctl restart ssh
```
*(Keep this terminal open, and test `ssh deployer@<SERVER_IP>` in a new tab before closing!)*

### 3. Configure UFW Firewall
```bash
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp comment 'SSH'
ufw allow 80/tcp comment 'HTTP (Caddy Let\'s Encrypt)'
ufw allow 443/tcp comment 'HTTPS (Caddy TLS)'
ufw enable
```

### 4. Enable Fail2ban & Automatic Security Updates
```bash
apt update && apt install -y fail2ban unattended-upgrades
systemctl enable --now fail2ban
dpkg-reconfigure -plow unattended-upgrades
```

### 5. Configure Swap (2 GB Safety Buffer)
To prevent Linux OOM-killer during peak document indexing:
```bash
fallocate -l 2G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
sysctl vm.swappiness=10
echo 'vm.swappiness=10' >> /etc/sysctl.d/99-swappiness.conf
```

### 6. Install Docker Engine & Compose
Install Docker from official Docker repository:
```bash
apt install -y ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt update
apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
usermod -aG docker deployer
```

---

## 🌐 Step 3: DNS Configuration

Log in to your DNS provider (Cloudflare, Namecheap, Route53, etc.) for `prince-gupta.me`:

| Type | Host / Name | Value | TTL | Proxy Status |
| :--- | :--- | :--- | :--- | :--- |
| **A** | `api` | `<YOUR_SERVER_IPV4>` | DNS Only (or 300s) | DNS Only (disable Cloudflare orange cloud initially for direct Caddy TLS) |
| **AAAA** | `api` | `<YOUR_SERVER_IPV6>` | DNS Only | *(Only add if your server has IPv6 assigned)* |

Verify DNS resolution from your workstation:
```bash
dig +short api.prince-gupta.me
# Should return your VPS IPv4
```

---

## 🚀 Step 4: Deploy InsureQuery

Log in as `deployer`:
```bash
ssh deployer@<SERVER_IP>
```

### 1. Clone the Repository
```bash
git clone https://github.com/Prince-5785/RAG-document-processor.git ~/insurequery
cd ~/insurequery
```

### 2. Configure Production Secrets in `.env`
Create `.env` with restrictive permissions:
```bash
touch .env
chmod 600 .env
```
Populate `.env`:
```ini
# Production Groq API Keys (comma-separated for automatic failover)
GROQ_API_KEYS="gsk_your_fresh_groq_api_key_here"

# Secure random token for API authentication (32 bytes hex)
# Generate with: openssl rand -hex 32
INSUREQUERY_API_KEY="your_generated_32_byte_hex_token"

# Frontend Origin for CORS
ALLOWED_ORIGINS="https://prince-gupta.me"

# Internal Docker Compose Redis
REDIS_URL="redis://redis:6379/0"

# Application Environment
APP_ENV="production"
```

### 3. Review `Caddyfile`
Verify that `Caddyfile` targets your domain:
```caddy
api.prince-gupta.me {
    request_body {
        max_size 30MB
    }

    # Hide internal observability counters from public callers
    @metrics path /metrics
    respond @metrics 404

    # Reverse proxy to InsureQuery FastAPI container with unbuffered SSE streaming
    reverse_proxy insurequery-api:8000 {
        flush_interval -1
    }
}
```

### 4. Build and Start Services
```bash
docker compose up -d --build
```
*Note: The initial build downloads PyTorch and caches the embedding (`all-MiniLM-L6-v2`) and cross-encoder (`ms-marco-MiniLM-L-6-v2`) models, taking ~2-3 minutes. Subsequent restarts are instantaneous.*

---

## 🔍 Step 5: Verification & Smoke Testing

Run these checks from your local laptop:

### 1. Port Exposure Audit
Verify that Redis and FastAPI are NOT exposed to the internet:
```bash
nmap -p 6379,8000 <SERVER_IP>
```
Expected output:
```
PORT     STATE    SERVICE
6379/tcp filtered redis
8000/tcp filtered unknown
```

### 2. Health Check
```bash
curl -i https://api.prince-gupta.me/health
```
Expected response: `HTTP/2 200` with `{"pipeline_status": "healthy", ...}`.

### 3. Fail-Closed Auth Test
Without credentials:
```bash
curl -i -X POST https://api.prince-gupta.me/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the room rent limit?"}'
```
Expected response: `HTTP/2 401 Unauthorized` (`{"detail": "Not authenticated"}`).

### 4. Metrics Protection Test
```bash
curl -i https://api.prince-gupta.me/metrics
```
Expected response: `HTTP/2 404 Not Found`.

### 5. Authenticated Query Test
```bash
API_KEY="<your_configured_key>"

curl -X POST https://api.prince-gupta.me/query \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the waiting period for pre-existing diseases?"}'
```

### 6. Streaming SSE Response Test
```bash
curl -N -X POST https://api.prince-gupta.me/query/stream \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key policy exclusions?"}'
```
Tokens should appear in real time on your terminal.

---

## 🛠️ Step 6: Day-2 Maintenance & Operations

### Viewing Live Logs
```bash
# All services
docker compose logs -f

# API container only
docker compose logs -f insurequery-api

# Caddy TLS and access logs
docker compose logs -f caddy
```

### Updating to Latest Code
```bash
cd ~/insurequery
git pull origin main
docker compose up -d --build
```

### Backing up Vector Store & Document Registry
The ChromaDB index and document registry persist in Docker volume `insure_data`:
```bash
# Create a compressed backup of the volume
docker run --rm \
  -v insurequery_insure_data:/volume \
  -v ~/backups:/backup \
  alpine tar -czf /backup/insure_data_$(date +%Y%m%d_%H%M%S).tar.gz -C /volume .
```

---

## 👥 Sharing with Recruiters & Portfolio Setup

- **Swagger Documentation**: Link directly to `https://api.prince-gupta.me/docs` in your portfolio and resume.
- **API Access**: Provide the Bearer key upon request. Never hardcode the key in client-side frontends.
- **Public Demo**: For a public web interface on `https://prince-gupta.me`, run a lightweight backend server (or Next.js API route) that signs requests with the server-side `INSUREQUERY_API_KEY` and applies rate-limiting (e.g. 5 requests/min per IP).
