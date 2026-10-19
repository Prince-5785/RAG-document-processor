"""
Hybrid Retriever combining BM25 Sparse Lexical Search with Dense Semantic Vector Search.
Uses Reciprocal Rank Fusion (RRF) to merge ranks, ensuring exact insurance terminology
(e.g., 'sub-limit', 'waiting period', 'co-pay', 'pre-existing disease') is reliably retrieved.
"""

import re
import logging
from typing import List, Dict, Any, Optional

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logging.warning("rank-bm25 not available. BM25 search will use token-overlap fallback.")


def simple_tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    if not text:
        return []
    return re.findall(r'\b\w+\b', text.lower())


class HybridRetriever:
    """Manages BM25 index and blends dense vector rankings with sparse keyword rankings."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('retrieval', {})
        hybrid_config = self.config.get('hybrid_search', {})
        self.enabled = hybrid_config.get('enabled', True)
        self.vector_weight = hybrid_config.get('vector_weight', 0.6)
        self.bm25_weight = hybrid_config.get('bm25_weight', 0.4)
        self.rrf_k = hybrid_config.get('rrf_k', 60)
        self.logger = logging.getLogger(__name__)

        self.corpus_chunks: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25_model: Optional[Any] = None

    def build_bm25_index(self, chunks: List[Dict[str, Any]]) -> None:
        """Build or rebuild BM25 index from a list of chunk dictionaries."""
        self.corpus_chunks = chunks
        self.tokenized_corpus = [simple_tokenize(chunk.get('text', '')) for chunk in chunks]

        if BM25_AVAILABLE and self.tokenized_corpus:
            try:
                self.bm25_model = BM25Okapi(self.tokenized_corpus)
                self.logger.info(f"Built BM25 index for {len(chunks)} chunks.")
            except Exception as e:
                self.logger.error(f"Failed to build BM25 index: {e}")
                self.bm25_model = None
        else:
            self.bm25_model = None

    def add_chunks_to_index(self, new_chunks: List[Dict[str, Any]]) -> None:
        """Add new chunks to the BM25 index."""
        if not new_chunks:
            return
        combined = list(self.corpus_chunks) + list(new_chunks)
        self.build_bm25_index(combined)

    def search_bm25(self, query: str, top_k: int = 25, document_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Execute BM25 lexical search. Returns top_k documents matching the query.
        """
        if not self.corpus_chunks or not query.strip():
            return []

        tokenized_query = simple_tokenize(query)
        if not tokenized_query:
            return []

        scores: List[float] = []
        if self.bm25_model and BM25_AVAILABLE:
            raw_scores = self.bm25_model.get_scores(tokenized_query)
            scores = [float(s) for s in raw_scores]
        else:
            # Fallback simple token overlap scoring
            q_set = set(tokenized_query)
            for tokens in self.tokenized_corpus:
                if not tokens:
                    scores.append(0.0)
                else:
                    overlap = len(q_set.intersection(tokens))
                    scores.append(float(overlap) / (len(q_set) + 0.1))

        # Build candidate list with document_id filtering if requested
        candidates = []
        for i, (chunk, score) in enumerate(zip(self.corpus_chunks, scores)):
            meta = chunk.get('metadata', {})
            if document_ids:
                doc_id = meta.get('document_id') or meta.get('file_hash')
                if doc_id and doc_id not in document_ids:
                    continue

            candidates.append({
                'text': chunk.get('text', ''),
                'metadata': meta,
                'bm25_score': score,
                'chunk_ref_id': i
            })

        candidates.sort(key=lambda x: x['bm25_score'], reverse=True)
        return candidates[:top_k]

    def fuse_results(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        top_k: int = 25
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) combining dense vector and BM25 search rankings.
        Formula: score = vector_weight / (k + rank_dense) + bm25_weight / (k + rank_bm25)
        """
        if not self.enabled:
            return dense_results[:top_k]

        fused_scores: Dict[str, float] = {}
        doc_store: Dict[str, Dict[str, Any]] = {}

        def get_doc_key(doc: Dict[str, Any]) -> str:
            # Unique signature for chunk matching
            meta = doc.get('metadata', {})
            file_name = meta.get('file_name', '')
            chunk_idx = meta.get('chunk_index', '')
            text_prefix = doc.get('text', '')[:80].strip()
            return f"{file_name}:{chunk_idx}:{text_prefix}"

        # Score dense rank
        for rank, doc in enumerate(dense_results):
            key = get_doc_key(doc)
            doc_store[key] = doc
            fused_scores[key] = fused_scores.get(key, 0.0) + (self.vector_weight / (self.rrf_k + rank + 1))

        # Score BM25 rank
        for rank, doc in enumerate(sparse_results):
            key = get_doc_key(doc)
            if key not in doc_store:
                doc_store[key] = doc
            fused_scores[key] = fused_scores.get(key, 0.0) + (self.bm25_weight / (self.rrf_k + rank + 1))

        # Rank all fused documents
        sorted_keys = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)
        results = []
        for key in sorted_keys[:top_k]:
            doc = dict(doc_store[key])
            doc['fused_score'] = fused_scores[key]
            # Carry over dense vector score if present
            if 'score' not in doc:
                doc['score'] = fused_scores[key]
            results.append(doc)

        return results
