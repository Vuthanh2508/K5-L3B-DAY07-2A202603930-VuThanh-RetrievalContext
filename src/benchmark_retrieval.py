"""Benchmark script for evaluating 5 retrieval configurations on Dev (and Test).

Evaluates:
1. Baseline: Dense Qdrant (bge-m3), top-3, budget 7200
2. Method 2: Lexical BM25Okapi, top-3, budget 7200
3. Method 3: Hybrid RRF (Dense + BM25), top-3, budget 7200
4. Improvement 1: Hybrid Z-Score Fusion (alpha=0.6), top-3, budget 7200
5. Improvement 2: Dense + Cross-Encoder Reranker (bge-reranker-v2-m3), top-3, budget 7200

Metrics:
- Hit@1: Number of queries with valid evidence in top-1 / total answerable queries
- Evidence Hit@3: Number of queries with valid evidence in top-3 / total answerable queries
- Context Evidence Rate: Number of queries with valid evidence packed into context / total answerable queries
- Pure query latency (ms/query) and estimated service cost ($0 on local GPU/Qdrant)
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRRFRetriever,
    RerankerRetriever,
    ZScoreHybridRetriever,
    hit,
    pack_context,
)


def run_benchmark(split: str = "dev", budget: int = 7200, top_k: int = 3):
    chunks_path = ROOT / "data" / "chunks.json"
    questions_path = ROOT / "data" / "questions.json"

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(questions_path, "r", encoding="utf-8") as f:
        all_q = json.load(f)

    target_q = [q for q in all_q if q.get("split") == split]
    print(f"=== Running Retrieval Benchmark on Split: '{split}' (N = {len(target_q)} questions, budget = {budget}) ===")

    print("Initializing retrievers...")
    dense = DenseRetriever(chunks, qdrant_url="http://localhost:6333", collection_name="collection_strategy_e")
    bm25 = BM25Retriever(chunks)
    hybrid_rrf = HybridRRFRetriever(dense, bm25, candidate_k=15, rrf_k=60)
    hybrid_zscore = ZScoreHybridRetriever(dense, bm25, alpha=0.6, candidate_k=15)
    reranker = RerankerRetriever(dense, candidate_k=15, reranker_model="BAAI/bge-reranker-v2-m3")

    # Warmup
    print("Warming up models on GPU...")
    dense.search("Netflix revenue warmup", top_k=1)
    bm25.search("Netflix revenue warmup", top_k=1)
    hybrid_rrf.search("Netflix revenue warmup", top_k=1)
    hybrid_zscore.search("Netflix revenue warmup", top_k=1)
    reranker.search("Netflix revenue warmup", top_k=1)
    print("Warmup complete. Starting benchmark.\n")

    configs = [
        ("Baseline (Dense Qdrant bge-m3)", dense),
        ("Phương pháp 2 (Lexical BM25Okapi)", bm25),
        ("Phương pháp 3 (Hybrid RRF)", hybrid_rrf),
        ("Cải thiện 1 (Hybrid Z-Score a=0.6)", hybrid_zscore),
        ("Cải thiện 2 (Dense + Reranker v2-m3)", reranker),
    ]

    summary_rows = []

    for name, retriever in configs:
        h1 = 0
        h3 = 0
        ctx_hits = 0
        latencies = []
        traces = []

        start_total = time.perf_counter()
        for q in target_q:
            t0 = time.perf_counter()
            results = retriever.search(q["question"], filters=q.get("filter"), top_k=top_k)
            t_query = (time.perf_counter() - t0) * 1000
            latencies.append(t_query)

            # Check hits
            is_h1 = bool(results and hit(results[0], q["gold"]))
            is_h3 = any(hit(r, q["gold"]) for r in results[:3])
            context_str, context_ids = pack_context(results, budget=budget)
            is_ctx = any(hit(dense.chunks[cid], q["gold"]) for cid in context_ids if cid in dense.chunks)

            if is_h1: h1 += 1
            if is_h3: h3 += 1
            if is_ctx: ctx_hits += 1

            traces.append({
                "id": q["id"],
                "question": q["question"],
                "h1": is_h1,
                "h3": is_h3,
                "ctx": is_ctx,
                "results": [{"id": r["id"], "score": r["score"], "hit": hit(r, q["gold"])} for r in results]
            })

        total_time = time.perf_counter() - start_total
        avg_ms = statistics.mean(latencies)
        med_ms = statistics.median(latencies)
        n = len(target_q)

        row = {
            "name": name,
            "split": split,
            "n": n,
            "h1": f"{h1}/{n} ({h1/n*100:.1f}%)",
            "h3": f"{h3}/{n} ({h3/n*100:.1f}%)",
            "ctx": f"{ctx_hits}/{n} ({ctx_hits/n*100:.1f}%)",
            "time_str": f"{total_time:.3f}s ({avg_ms:.1f}ms/q)",
            "traces": traces
        }
        summary_rows.append(row)

    print("=" * 120)
    print(f"{'Cấu hình':38s} | {'Split':5s} | {'Hit@1':15s} | {'Evidence Hit@3':16s} | {'Context Evidence':17s} | {'Thời gian / chi phí':20s}")
    print("=" * 120)
    for r in summary_rows:
        print(f"{r['name']:38s} | {r['split']:5s} | {r['h1']:15s} | {r['h3']:16s} | {r['ctx']:17s} | {r['time_str']} / $0")
    print("=" * 120)

    # Save detailed traces for report audit
    out_dir = ROOT / "starter" / "results" / f"retrieval_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "retrieval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to: {out_dir / 'retrieval_summary.json'}")
    return summary_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--budget", type=int, default=7200)
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()
    run_benchmark(split=args.split, budget=args.budget, top_k=args.top_k)
