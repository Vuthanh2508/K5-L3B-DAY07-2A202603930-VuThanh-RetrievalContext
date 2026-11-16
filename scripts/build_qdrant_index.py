"""Script to build and populate Qdrant vector collection from data/chunks.json.

Uses BAAI/bge-m3 (Dense 1024-dim, Cosine distance) and upserts all 88 chunks
into collection 'collection_strategy_e'.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# CRITICAL FOR WINDOWS: Import torch/FlagEmbedding before httpx/openai
import torch
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


def build_index(chunks_path: Path, qdrant_url: str, collection_name: str, model_name: str = "BAAI/bge-m3"):
    print(f"=== Building Qdrant Index for Collection: '{collection_name}' ===")
    print(f"Chunks source: {chunks_path}")
    print(f"Qdrant URL   : {qdrant_url}")
    print(f"Model        : {model_name}")

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found at {chunks_path}")

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks.")

    print(f"Loading embedding model: {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = BGEM3FlagModel(model_name, use_fp16=(device == "cuda"))

    print(f"Connecting to Qdrant at {qdrant_url}...")
    client = QdrantClient(url=qdrant_url)

    collections = [c.name for c in client.get_collections().collections]
    if collection_name in collections:
        print(f"Collection '{collection_name}' already exists. Recreating for clean indexing...")
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )
    print(f"Created collection '{collection_name}' with 1024-dim Cosine vectors.")

    print("Generating dense embeddings for chunks...")
    texts = [c["text"] for c in chunks]
    t0 = time.perf_counter()
    embeddings = model.encode(texts, batch_size=16)["dense_vecs"]
    t_encode = time.perf_counter() - t0
    print(f"Encoded {len(texts)} chunks in {t_encode:.2f}s ({len(texts)/t_encode:.1f} chunks/s).")

    print("Upserting vectors into Qdrant...")
    points = []
    for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        meta = chunk.get("metadata", {})
        payload = {
            "id": chunk["id"],
            "source_id": chunk.get("source_id", "netflix-2023q2-10q"),
            "source": chunk.get("source", "NETFLIX_2023Q2_10Q-3-33.pdf"),
            "page": chunk.get("page", 0),
            "text": chunk.get("text", ""),
            "metadata": meta,
            "section": meta.get("section", "")
        }
        points.append(PointStruct(id=idx, vector=vector.tolist(), payload=payload))

    client.upsert(collection_name=collection_name, points=points)
    info = client.get_collection(collection_name)
    print(f"Successfully indexed {info.points_count} points in collection '{collection_name}'!")


def main():
    parser = argparse.ArgumentParser(description="Build Qdrant index for Strategy E chunks.")
    parser.add_argument("--chunks", type=Path, default=ROOT / "data" / "chunks.json")
    parser.add_argument("--qdrant_url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--collection", default=os.environ.get("QDRANT_COLLECTION", "collection_strategy_e"))
    parser.add_argument("--model", default="BAAI/bge-m3")
    args = parser.parse_args()

    build_index(args.chunks, args.qdrant_url, args.collection, args.model)


if __name__ == "__main__":
    main()
