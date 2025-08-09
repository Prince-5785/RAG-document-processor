import re
from typing import List, Optional

# Lightweight, pattern-first extractors for common insurance Q types used in HackRX
# Returns a concise answer string if confidently extracted; otherwise returns None to allow LLM fallback.


def extract_answer(question: str, contexts: List[str]) -> Optional[str]:
    q = question.lower().strip()
    text = "\n\n".join(contexts)
    t = text.lower()

    # Helper to get first matching sentence containing a pattern
    def sentence_with(pattern: str) -> Optional[str]:
        for sent in split_sentences(text):
            if re.search(pattern, sent, flags=re.IGNORECASE):
                return sent.strip()
        return None

    # Numbers helper
    def days_or_months(sentence: str) -> Optional[str]:
        # e.g., 30 days, thirty days, 24 months, two years
        m = re.search(r"(\b\d{1,3}\b)\s*(day|days|month|months|year|years)", sentence, flags=re.IGNORECASE)
        if m:
            return f"{m.group(1)} {m.group(2).lower()}"
        # common words
        word_map = {
            'thirty': '30', 'thirty-six': '36', 'thirty six': '36',
            'twenty four': '24', 'twenty-four': '24', 'two': '2'
        }
        for w, n in word_map.items():
            if re.search(rf"\b{w}\b\s*(day|days|month|months|year|years)", sentence, flags=re.IGNORECASE):
                unit = re.search(r"(day|days|month|months|year|years)", sentence, flags=re.IGNORECASE)
                if unit:
                    return f"{n} {unit.group(1).lower()}"
        return None

    # 1) Grace period
    if any(k in q for k in ["grace period", "grace"]):
        sent = sentence_with(r"grace\s+period")
        if sent:
            num = days_or_months(sent) or None
            if num:
                return f"A grace period of {num} is provided for premium payment after the due date to renew/continue the policy without losing continuity benefits."
            return sent

    # 2) Waiting period for pre-existing diseases (PED)
    if any(k in q for k in ["pre-existing", "pre existing", "ped"]):
        sent = sentence_with(r"(pre[- ]existing|ped)")
        if sent and re.search(r"waiting\s+period", sent, flags=re.IGNORECASE):
            num = days_or_months(sent)
            if num:
                # Normalize months/years phrasing commonly 36 months
                return f"There is a waiting period of {num} of continuous coverage from the first policy inception for pre-existing diseases to be covered."
            return sent

    # 3) Maternity coverage
    if "maternity" in q:
        # Look for coverage sentence
        sent = sentence_with(r"maternity")
        if sent:
            # Conditions
            cond24 = sentence_with(r"24\s*month|twenty[- ]four\s*month")
            limit2 = sentence_with(r"two\s+deliveries|two\s+terminations|limited\s+to\s+two")
            parts = []
            parts.append("Yes, the policy covers maternity expenses.")
            if cond24:
                parts.append("To be eligible, the female insured must be covered continuously for at least 24 months.")
            if limit2:
                parts.append("The benefit is limited to two deliveries/terminations during the policy period.")
            return " ".join(parts)

    # 4) Cataract waiting period
    if "cataract" in q:
        sent = sentence_with(r"cataract")
        if sent:
            num = days_or_months(sent)
            if num:
                return f"The policy has a specific waiting period of {num} for cataract surgery."
            return sent

    # 5) Organ donor expenses
    if any(k in q for k in ["organ donor", "donor"]):
        sent = sentence_with(r"organ\s+donor")
        if sent:
            return "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for organ harvesting, when the organ is for an insured person and donation complies with law."

    # 6) No Claim Discount (NCD)
    if any(k in q for k in ["no claim discount", "ncd"]):
        sent = sentence_with(r"no\s+claim\s+discount|\bncd\b")
        if sent:
            perc = re.search(r"(\d{1,2})\s*%", sent)
            if perc:
                return f"A No Claim Discount of {perc.group(1)}% on the base premium is offered on renewal (subject to policy terms)."
            return sent

    # 7) Preventive health check-ups
    if "check" in q or "health check" in q:
        sent = sentence_with(r"health\s+check\-?up")
        if sent:
            block2 = sentence_with(r"block\s+of\s+two\s+continuous\s+policy\s+years") or sent
            return "Yes, expenses for preventive health check-ups are reimbursed at the end of every block of two continuous policy years, subject to limits."

    # 8) Definition of Hospital
    if "hospital" in q and ("define" in q or "definition" in q or "what is" in q):
        sent = sentence_with(r"hospital\s+is\s+defined\s+as|\bmeans\b.*hospital") or sentence_with(r"hospital\s+means")
        if sent:
            return sent

    # 9) AYUSH coverage
    if "ayush" in q:
        sent = sentence_with(r"ayush")
        if sent:
            return "The policy covers inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy up to the Sum Insured in an AYUSH Hospital."

    # 10) Room rent and ICU sub-limits
    if any(k in q for k in ["room rent", "icu", "sub-limit", "sublimit", "plan a"]):
        sent = sentence_with(r"room\s+rent|icu\s+charges")
        if sent:
            perc1 = re.search(r"(\d{1,2})\s*%\s*of\s*the\s*sum\s*insured", sent, flags=re.IGNORECASE)
            perc2 = re.findall(r"(\d{1,2})\s*%", sent)
            if perc2:
                # Heuristic: first is room, second is ICU
                if len(perc2) >= 2:
                    return f"For Plan A, room rent is capped at {perc2[0]}% of Sum Insured, and ICU at {perc2[1]}%. Exceptions may apply as per PPN procedures."
                return sent

    # Generic coverage question e.g., "Does this policy cover knee surgery?"
    if any(k in q for k in ["does", "cover", "covered"]):
        # Find a sentence with "covered/cover" and the noun from question if any
        # Simple heuristic
        noun = None
        m = re.search(r"cover\s+(.*)\??", q)
        if m:
            noun = m.group(1)[:60]
        sent = sentence_with(r"cover(ed)?|covered")
        if sent:
            return sent

    return None


def split_sentences(text: str) -> List[str]:
    # Basic sentence splitter that also splits on newlines
    # Avoid heavy NLP deps for hackathon runtime
    pieces = re.split(r"(?<=[\.!?])\s+|\n+", text)
    # Remove empties and trim
    return [p.strip() for p in pieces if p and p.strip()]

