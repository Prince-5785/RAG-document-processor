# Deployment Guide

This guide covers different deployment options for the LLM-Powered Document Processing System.

## 🚀 Quick Start (Local Development)

### Prerequisites
- Python 3.8+
- 8GB+ RAM (for local LLM)
- GPU recommended (for faster inference)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd LLM-Powered-Document-Processing-System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 🐳 Docker Deployment

### Single Container
```bash
# Build the image
docker build -t rag-system .

# Run the container
docker run -p 8501:8501 -v $(pwd)/data:/app/data rag-system
```

### Docker Compose
```bash
# Start the services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the services
docker-compose down
```

### Production with Nginx
```bash
# Start with production profile
docker-compose --profile production up -d
```

## ☁️ Cloud Deployment

### AWS EC2

1. **Launch EC2 Instance**
   - Instance type: t3.large or larger (for LLM)
   - AMI: Ubuntu 20.04 LTS
   - Security group: Allow ports 22, 80, 8501

2. **Setup on EC2**
   ```bash
   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install Docker
   sudo apt install docker.io docker-compose -y
   sudo usermod -aG docker ubuntu
   
   # Clone and deploy
   git clone <repository-url>
   cd LLM-Powered-Document-Processing-System
   docker-compose up -d
   ```

3. **Configure Domain (Optional)**
   - Point your domain to EC2 public IP
   - Use Let's Encrypt for SSL

### Google Cloud Platform

1. **Create Compute Engine Instance**
   ```bash
   gcloud compute instances create rag-system \
     --machine-type=n1-standard-4 \
     --image-family=ubuntu-2004-lts \
     --image-project=ubuntu-os-cloud \
     --boot-disk-size=50GB
   ```

2. **Deploy Application**
   ```bash
   # SSH to instance
   gcloud compute ssh rag-system
   
   # Install dependencies and deploy
   sudo apt update
   sudo apt install docker.io docker-compose git -y
   git clone <repository-url>
   cd LLM-Powered-Document-Processing-System
   sudo docker-compose up -d
   ```

### Azure Container Instances

1. **Create Resource Group**
   ```bash
   az group create --name rag-system-rg --location eastus
   ```

2. **Deploy Container**
   ```bash
   az container create \
     --resource-group rag-system-rg \
     --name rag-system \
     --image <your-registry>/rag-system:latest \
     --ports 8501 \
     --memory 8 \
     --cpu 4
   ```

## 🔧 Configuration

### Environment Variables
```bash
# Model configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=microsoft/DialoGPT-medium
DEVICE=cpu  # or cuda for GPU

# Storage configuration
CHROMA_PERSIST_DIR=/app/data/chroma_index
CACHE_DIR=/app/data/cache

# UI configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Custom Configuration
Edit `config/config.yaml` to customize:
- Model settings
- Chunk sizes
- Retrieval parameters
- UI preferences

## 📊 Monitoring

### Health Checks
```bash
# Check application health
curl http://localhost:8501/_stcore/health

# Check container status
docker ps
docker logs <container-id>
```

### Logs
```bash
# View application logs
docker-compose logs -f rag-system

# View specific service logs
docker logs <container-name>
```

### Metrics
- Monitor CPU/Memory usage
- Track response times
- Monitor disk space for vector store

## 🔒 Security

### Production Security Checklist
- [ ] Use HTTPS (SSL/TLS)
- [ ] Implement authentication
- [ ] Restrict file upload types
- [ ] Set up firewall rules
- [ ] Regular security updates
- [ ] Monitor access logs

### Authentication (Optional)
Add authentication to Streamlit:
```python
# In app.py
import streamlit_authenticator as stauth

# Configure authentication
authenticator = stauth.Authenticate(...)
name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    # Main application code
    pass
elif authentication_status == False:
    st.error('Username/password is incorrect')
```

## 🚀 Performance Optimization

### GPU Acceleration
```yaml
# docker-compose.yml
services:
  rag-system:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Caching
- Enable embedding cache in config
- Use Redis for distributed caching
- Implement query result caching

### Scaling
- Use load balancer for multiple instances
- Implement horizontal scaling
- Consider using managed services for vector store

## 🔄 Updates and Maintenance

### Update Process
```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Backup
```bash
# Backup vector store
tar -czf chroma_backup_$(date +%Y%m%d).tar.gz data/chroma_index/

# Backup configuration
cp -r config/ config_backup_$(date +%Y%m%d)/
```

### Restore
```bash
# Restore vector store
tar -xzf chroma_backup_YYYYMMDD.tar.gz -C data/

# Restart services
docker-compose restart
```

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**
   - Increase container memory limits
   - Use smaller models
   - Enable model quantization

2. **Slow Performance**
   - Use GPU acceleration
   - Optimize chunk sizes
   - Enable caching

3. **Model Loading Errors**
   - Check internet connection
   - Verify model names
   - Check disk space

4. **Vector Store Issues**
   - Check permissions on data directory
   - Verify ChromaDB installation
   - Reset index if corrupted

### Debug Mode
```bash
# Run with debug logging
STREAMLIT_LOGGER_LEVEL=debug streamlit run app.py

# Check system status
python -c "from src.rag_pipeline import RAGPipeline; p = RAGPipeline(); print(p.get_system_status())"
```

## 📞 Support

For deployment issues:
1. Check logs first
2. Review configuration
3. Test with sample data
4. Contact support with error details

## 📚 Additional Resources

- [Streamlit Deployment Guide](https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
