"""Core Retrieval Module for Day 7 — Agentic RAG.
Connects directly to Day 6 Qdrant Vector Store (collection_strategy_e) + BAAI/bge-m3.

Implements:
1. Dense Retrieval using BAAI/bge-m3 + Qdrant Docker Server (localhost:6333)
2. Lexical Retrieval using BM25Okapi (rank-bm25)
3. Hybrid Retrieval using Reciprocal Rank Fusion (RRF)
4. Improvement 1: Hybrid Z-Score Fusion (0.6 Dense + 0.4 BM25)
5. Improvement 2: Retrieval + Cross-Encoder Reranker (BAAI/bge-reranker-v2-m3)
6. Context Packing with Deduplication and Character Budget (7,200 chars)
"""
import math
import os
import re
import statistics
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from FlagEmbedding import BGEM3FlagModel, FlagReranker
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words with NFC unicode normalization."""
    return re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold())


def hit(chunk: Dict[str, Any], gold: List[Dict[str, Any]]) -> bool:
    """Return True if chunk matches any gold evidence in source, page, and exact quote."""
    chunk_source = chunk.get('source_id') or chunk.get('source')
    chunk_page = chunk.get('page')
    chunk_text = chunk.get('text', '')
    for g in gold:
        # Match source if provided
        g_src = g.get('source_id') or g.get('source')
        if g_src and chunk_source and (g_src not in chunk_source and chunk_source not in g_src):
            continue
        # Match page
        if g.get('page') != chunk_page:
            continue
        # Match evidence text
        if g.get('evidence', '') in chunk_text:
            return True
    return False


def pack_context(results: List[Dict[str, Any]], budget: int = 7200) -> Tuple[str, List[str]]:
    """Pack top retrieved chunks into context string with deduplication and budget ceiling.
    
    Default budget is 7,200 chars based on worst-case 3 max-length chunks (3 * 2350 ~ 7050 + headers).
    """
    blocks, ids, seen = [], [], set()
    for c in results:
        cid = c['id']
        if cid in seen:
            continue
        seen.add(cid)
        source = c.get('source', 'NETFLIX_2023Q2_10Q-3-33.pdf')
        page = c.get('page', 0)
        block = f"[{cid} | {source} | page {page}]\n{c['text']}"
        candidate_context = '\n\n'.join(blocks + [block])
        if len(candidate_context) <= budget:
            blocks.append(block)
            ids.append(cid)
    return '\n\n'.join(blocks), ids


class DenseRetriever:
    """Dense vector retriever using BAAI/bge-m3 query embeddings and Qdrant collection."""
    def __init__(self, chunks: List[Dict[str, Any]], qdrant_url: str = "http://localhost:6333",
                 collection_name: str = "collection_strategy_e", model_name: str = "BAAI/bge-m3"):
        self.chunks = {c['id']: c for c in chunks}
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.model = BGEM3FlagModel(model_name, use_fp16=True)

    def search(self, query: str, *, filters: Optional[Dict[str, Any]] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        # Compute dense query vector
        embed_res = self.model.encode(query, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        dense_vec = embed_res['dense_vecs'].tolist()

        # Build Qdrant filter if provided
        qdrant_filter = None
        if filters:
            must_conditions = []
            for k, v in filters.items():
                must_conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))
            if must_conditions:
                qdrant_filter = Filter(must=must_conditions)

        # Query Qdrant
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vec,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True
        )

        results = []
        for point in response.points:
            cid = point.payload.get('chunk_id') or point.payload.get('id') or str(point.id)
            chunk_data = self.chunks.get(cid, {})
            results.append({
                'id': cid,
                'source_id': chunk_data.get('source_id', point.payload.get('document_id')),
                'source': chunk_data.get('source', point.payload.get('source')),
                'page': chunk_data.get('page', point.payload.get('page')),
                'text': chunk_data.get('text', point.payload.get('text', '')),
                'metadata': chunk_data.get('metadata', point.payload.get('metadata', {})),
                'score': point.score,
                'strategy': 'dense'
            })
        return results


class BM25Retriever:
    """Lexical keyword retriever using BM25Okapi over tokenized chunks."""
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks_list = chunks
        self.tokenized_corpus = [tokenize(c['text']) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, *, filters: Optional[Dict[str, Any]] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        scored_chunks = []
        for c, score in zip(self.chunks_list, scores):
            if filters:
                match = True
                for fk, fv in filters.items():
                    if c.get(fk) != fv and c.get('metadata', {}).get(fk) != fv:
                        match = False
                        break
                if not match:
                    continue
            if score > 0:
                scored_chunks.append({
                    'id': c['id'],
                    'source_id': c['source_id'],
                    'source': c['source'],
                    'page': c['page'],
                    'text': c['text'],
                    'metadata': c['metadata'],
                    'score': float(score),
                    'strategy': 'bm25'
                })
        scored_chunks.sort(key=lambda x: (-x['score'], x['id']))
        return scored_chunks[:top_k]


class HybridRRFRetriever:
    """Hybrid Retriever combining Dense and BM25 via Reciprocal Rank Fusion (RRF)."""
    def __init__(self, dense_retriever: DenseRetriever, bm25_retriever: BM25Retriever,
                 candidate_k: int = 15, rrf_k: int = 60):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def search(self, query: str, *, filters: Optional[Dict[str, Any]] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        dense_results = self.dense.search(query, filters=filters, top_k=self.candidate_k)
        bm25_results = self.bm25.search(query, filters=filters, top_k=self.candidate_k)

        all_candidates: Dict[str, Dict[str, Any]] = {}
        rrf_scores: Dict[str, float] = {}

        for rank, item in enumerate(dense_results):
            cid = item['id']
            all_candidates[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        for rank, item in enumerate(bm25_results):
            cid = item['id']
            all_candidates[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: (-rrf_scores[cid], cid))
        results = []
        for cid in sorted_ids[:top_k]:
            item = dict(all_candidates[cid])
            item['score'] = rrf_scores[cid]
            item['strategy'] = 'hybrid_rrf'
            results.append(item)
        return results


class ZScoreHybridRetriever:
    """Improvement 1: Hybrid Retriever combining Dense and BM25 via Z-score normalization."""
    def __init__(self, dense_retriever: DenseRetriever, bm25_retriever: BM25Retriever,
                 alpha: float = 0.6, candidate_k: int = 15):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.alpha = alpha  # Weight for dense (1 - alpha for BM25)
        self.candidate_k = candidate_k

    def _normalize_zscore(self, items: List[Dict[str, Any]]) -> Dict[str, float]:
        if not items:
            return {}
        scores = [item['score'] for item in items]
        if len(scores) == 1 or statistics.stdev(scores) == 0:
            return {item['id']: 1.0 for item in items}
        mean = statistics.mean(scores)
        stdev = statistics.stdev(scores)
        return {item['id']: (item['score'] - mean) / stdev for item in items}

    def search(self, query: str, *, filters: Optional[Dict[str, Any]] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        dense_results = self.dense.search(query, filters=filters, top_k=self.candidate_k)
        bm25_results = self.bm25.search(query, filters=filters, top_k=self.candidate_k)

        dense_z = self._normalize_zscore(dense_results)
        bm25_z = self._normalize_zscore(bm25_results)

        all_candidates: Dict[str, Dict[str, Any]] = {}
        for r in dense_results:
            all_candidates[r['id']] = r
        for r in bm25_results:
            if r['id'] not in all_candidates:
                all_candidates[r['id']] = r

        combined_scores = {}
        for cid in all_candidates:
            dz = dense_z.get(cid, -2.0)
            bz = bm25_z.get(cid, -2.0)
            combined_scores[cid] = self.alpha * dz + (1.0 - self.alpha) * bz

        sorted_ids = sorted(combined_scores.keys(), key=lambda cid: (-combined_scores[cid], cid))
        results = []
        for cid in sorted_ids[:top_k]:
            item = dict(all_candidates[cid])
            item['score'] = combined_scores[cid]
            item['strategy'] = 'hybrid_zscore'
            results.append(item)
        return results


class RerankerRetriever:
    """Improvement 2: Dense Retrieval + Cross-Encoder Reranker (bge-reranker-v2-m3)."""
    def __init__(self, dense_retriever: DenseRetriever, candidate_k: int = 15,
                 reranker_model: str = "BAAI/bge-reranker-v2-m3"):
        self.dense = dense_retriever
        self.candidate_k = candidate_k
        self.reranker = FlagReranker(reranker_model, use_fp16=True)

    def search(self, query: str, *, filters: Optional[Dict[str, Any]] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        candidates = self.dense.search(query, filters=filters, top_k=self.candidate_k)
        if not candidates:
            return []
        # Score query-document pairs with cross-encoder
        pairs = [[query, c['text']] for c in candidates]
        scores = self.reranker.compute_score(pairs)
        if isinstance(scores, (int, float)):
            scores = [scores]
        for c, s in zip(candidates, scores):
            c['score'] = float(s)
            c['strategy'] = 'dense_reranker'
        candidates.sort(key=lambda x: (-x['score'], x['id']))
        return candidates[:top_k]
