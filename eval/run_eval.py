"""
InsureQuery Retrieval Evaluation Suite.
Evaluates retrieval accuracy across configurations:
- Baseline Dense-Only Retrieval (Vector search without reranking)
- Dense + Cross-Encoder Re-ranking
- InsureQuery Hybrid Retrieval (Dense Vector + BM25 Lexical + Cross-Encoder + Threshold Guardrail)

Computes Hit@1, Hit@3, Hit@5, MRR (Mean Reciprocal Rank), and calculates exact accuracy improvements.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_pipeline import RAGPipeline
from src.utils import load_json, save_json


def check_hit(retrieved_chunks: List[Dict[str, Any]], ground_truth_keywords: List[str], top_k: int) -> bool:
    """Checks if any retrieved chunk within top_k contains the ground truth keywords."""
    for chunk in retrieved_chunks[:top_k]:
        text = chunk.get('text', '').lower()
        # All keywords in list must be present in chunk
        if all(kw.lower() in text for kw in ground_truth_keywords):
            return True
    return False


def compute_reciprocal_rank(retrieved_chunks: List[Dict[str, Any]], ground_truth_keywords: List[str]) -> float:
    """Computes reciprocal rank (1 / rank) for the first matching chunk."""
    for rank, chunk in enumerate(retrieved_chunks, 1):
        text = chunk.get('text', '').lower()
        if all(kw.lower() in text for kw in ground_truth_keywords):
            return 1.0 / rank
    return 0.0


def run_evaluation():
    print("=" * 70)
    print(" InsureQuery: RAG Retrieval Accuracy Benchmark")
    print("=" * 70)

    # 1. Initialize Pipeline
    pipeline = RAGPipeline()
    pipeline.reset_index()

    # 2. Index evaluation documents
    doc_paths = [
        str(PROJECT_ROOT / "data" / "documents" / "sample_insurance_policy.txt"),
        str(PROJECT_ROOT / "data" / "documents" / "claims_guidelines.txt")
    ]
    print(f"\n[1/3] Indexing evaluation documents: {[os.path.basename(p) for p in doc_paths]}")
    index_res = pipeline.index_documents(doc_paths, force_reindex=True)
    print(f"      Indexed {index_res['processed_files']} files ({index_res['total_chunks']} chunks).")

    # 3. Load QA dataset
    qa_path = PROJECT_ROOT / "eval" / "qa_set.json"
    qa_set = load_json(str(qa_path))
    total_q = len(qa_set)
    print(f"\n[2/3] Loaded {total_q} hand-labeled QA pairs from {qa_path.name}")

    configs = [
        {
            'name': 'Baseline: Dense Vector Only',
            'use_bm25': False,
            'use_reranker': False,
            'threshold': 0.0
        },
        {
            'name': 'Dense + Cross-Encoder Reranker',
            'use_bm25': False,
            'use_reranker': True,
            'threshold': 0.0
        },
        {
            'name': 'InsureQuery Hybrid (BM25 + Dense + Reranker + Guardrail)',
            'use_bm25': True,
            'use_reranker': True,
            'threshold': 0.3
        }
    ]

    all_results = {}

    print("\n[3/3] Running evaluations across configurations...")
    for cfg in configs:
        cfg_name = cfg['name']
        print(f"\n  Evaluating: {cfg_name}")

        hit_1_count = 0
        hit_3_count = 0
        hit_5_count = 0
        mrr_total = 0.0
        latencies = []

        for item in qa_set:
            q = item['question']
            keywords = item['ground_truth_keywords']

            t0 = time.time()
            
            # Retrieval logic according to config
            if not cfg['use_bm25'] and not cfg['use_reranker']:
                # Baseline dense vector search only
                q_emb = pipeline.embedding_service.encode_single_text(q)
                retrieved = pipeline.vector_store.similarity_search(q_emb, top_k=5)
            elif not cfg['use_bm25'] and cfg['use_reranker']:
                # Dense + reranker
                q_emb = pipeline.embedding_service.encode_single_text(q)
                candidates = pipeline.vector_store.similarity_search(q_emb, top_k=pipeline.candidate_k)
                retrieved = pipeline.embedding_service.rerank_documents(q, candidates)[:5]
            else:
                # Hybrid BM25 + Dense + Reranker + Threshold Guardrail
                q_emb = pipeline.embedding_service.encode_single_text(q)
                dense_candidates = pipeline.vector_store.similarity_search(q_emb, top_k=pipeline.candidate_k)
                bm25_candidates = pipeline.hybrid_retriever.search_bm25(q, top_k=pipeline.candidate_k)
                fused = pipeline.hybrid_retriever.fuse_results(dense_candidates, bm25_candidates, top_k=pipeline.candidate_k)
                reranked = pipeline.embedding_service.rerank_documents(q, fused)
                
                # Apply threshold guardrail
                if reranked and reranked[0].get('rerank_score', 0) >= cfg['threshold']:
                    retrieved = [doc for doc in reranked[:5] if doc.get('rerank_score', 0) >= cfg['threshold']]
                else:
                    retrieved = []

            latencies.append((time.time() - t0) * 1000)

            # Score hits
            if check_hit(retrieved, keywords, top_k=1):
                hit_1_count += 1
            if check_hit(retrieved, keywords, top_k=3):
                hit_3_count += 1
            if check_hit(retrieved, keywords, top_k=5):
                hit_5_count += 1
            mrr_total += compute_reciprocal_rank(retrieved, keywords)

        hit_1_pct = (hit_1_count / total_q) * 100.0
        hit_3_pct = (hit_3_count / total_q) * 100.0
        hit_5_pct = (hit_5_count / total_q) * 100.0
        mrr_score = mrr_total / total_q
        avg_lat = sum(latencies) / len(latencies)

        all_results[cfg_name] = {
            'hit_1': round(hit_1_pct, 2),
            'hit_3': round(hit_3_pct, 2),
            'hit_5': round(hit_5_pct, 2),
            'mrr': round(mrr_score, 4),
            'avg_latency_ms': round(avg_lat, 2)
        }
        print(f"    -> Hit@1: {hit_1_pct:.1f}% | Hit@3: {hit_3_pct:.1f}% | Hit@5: {hit_5_pct:.1f}% | MRR: {mrr_score:.4f} | Latency: {avg_lat:.1f}ms")

    # Calculate improvement of Hybrid over Baseline
    base = all_results['Baseline: Dense Vector Only']
    hybrid = all_results['InsureQuery Hybrid (BM25 + Dense + Reranker + Guardrail)']
    hit1_diff = hybrid['hit_1'] - base['hit_1']
    hit1_relative = ((hybrid['hit_1'] - base['hit_1']) / (base['hit_1'] or 1.0)) * 100.0
    mrr_relative = ((hybrid['mrr'] - base['mrr']) / (base['mrr'] or 1.0)) * 100.0

    print("\n" + "=" * 70)
    print(" EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"| {'Configuration':<52} | {'Hit@1':<7} | {'Hit@3':<7} | {'Hit@5':<7} | {'MRR':<7} |")
    print(f"|{'-'*54}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*9}|")
    for name, m in all_results.items():
        print(f"| {name:<52} | {m['hit_1']:>5.1f}% | {m['hit_3']:>5.1f}% | {m['hit_5']:>5.1f}% | {m['mrr']:>7.4f} |")
    print(f"\nNet Accuracy Gain (Hit@1): +{hit1_diff:.1f}% absolute (+{hit1_relative:.1f}% relative)")
    print(f"Net MRR Improvement:      +{mrr_relative:.1f}% relative")
    print("=" * 70)

    # Save to JSON
    output_path = PROJECT_ROOT / "eval" / "eval_results.json"
    save_json({
        'total_evaluated_queries': total_q,
        'results': all_results,
        'improvement': {
            'hit_1_absolute_gain': round(hit1_diff, 2),
            'hit_1_relative_gain_pct': round(hit1_relative, 2),
            'mrr_relative_gain_pct': round(mrr_relative, 2)
        }
    }, str(output_path))
    print(f"\nSaved evaluation metrics to: {output_path}")

    return all_results


if __name__ == "__main__":
    run_evaluation()
