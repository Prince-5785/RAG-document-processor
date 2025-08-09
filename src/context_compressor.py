import re
from typing import List, Dict, Any
import numpy as np

class ContextCompressor:
    """Extractive, sentence-level context compression to reduce tokens.
    Selects top relevant sentences from reranked chunks based on cosine similarity to the query.
    """

    def __init__(self, embedding_service, max_sentences: int = 8, per_doc_max: int = 3):
        self.embedding_service = embedding_service
        self.max_sentences = max_sentences
        self.per_doc_max = per_doc_max

    def _split_sentences(self, text: str) -> List[str]:
        # simple splitter: split on ., !, ?, and newlines; keep short sentences filtered out later
        parts = re.split(r"(?<=[\.!?])\s+|\n+", text)
        return [p.strip() for p in parts if p and len(p.strip()) > 20]

    def _clean_markers(self, s: str) -> str:
        # remove heading markers we inserted like === ... ===
        s = re.sub(r"^===\s*[^=]+\s*===\s*$", "", s, flags=re.IGNORECASE | re.MULTILINE).strip()
        s = s.replace("===", " ")
        return re.sub(r"\s+", " ", s).strip()

    def compress(self, query: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not contexts:
            return []
        # Gather sentences per doc with metadata
        all_sentences: List[Dict[str, Any]] = []
        for ctx in contexts:
            text = self._clean_markers(ctx.get('text', ''))
            sentences = self._split_sentences(text)
            for sent in sentences[:100]:  # cap per chunk for speed
                all_sentences.append({
                    'text': sent,
                    'metadata': ctx.get('metadata', {})
                })
        if not all_sentences:
            return contexts
        # Embed query and sentences
        qvec = self.embedding_service.encode_single_text(query)
        sent_texts = [s['text'] for s in all_sentences]
        s_emb = self.embedding_service.encode_texts(sent_texts, show_progress=False)
        s_emb = np.array(s_emb)
        if qvec.size == 0:
            # no query vec; return first few sentences grouped
            chosen = all_sentences[: self.max_sentences]
        else:
            # score by cosine similarity
            def cos_sim(a, b):
                return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
            scores = [cos_sim(qvec, s_emb[i]) for i in range(len(all_sentences))]
            # sort by score desc
            ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            # per-doc cap
            per_doc_counts: Dict[str, int] = {}
            chosen = []
            for i in ranked_idx:
                meta = all_sentences[i]['metadata'] or {}
                doc_id = meta.get('file_hash') or meta.get('file_name') or 'doc'
                if per_doc_counts.get(doc_id, 0) >= self.per_doc_max:
                    continue
                chosen.append(all_sentences[i])
                per_doc_counts[doc_id] = per_doc_counts.get(doc_id, 0) + 1
                if len(chosen) >= self.max_sentences:
                    break
        # Merge chosen sentences into 2-3 blocks to keep structure
        blocks: List[Dict[str, Any]] = []
        block_size = max(2, self.max_sentences // 2)
        for i in range(0, len(chosen), block_size):
            block_sents = [c['text'] for c in chosen[i:i+block_size]]
            if not block_sents:
                continue
            # Prefer metadata of first sentence in block
            meta = chosen[i].get('metadata', {}) if i < len(chosen) else {}
            blocks.append({'text': " \n".join(block_sents), 'metadata': meta})
        return blocks if blocks else contexts

