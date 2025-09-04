#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any

from pymilvus import MilvusClient
from backend.config import DB_PATH, COLLECTION, VECTOR_FIELD, DIMENSION, SERVICE_URL, TOP_K_DEFAULT

def embed_query(query: str) -> np.ndarray:
    resp = requests.post(f"{SERVICE_URL}/embed", json={"texts": [query]}, timeout=60)
    resp.raise_for_status()
    vecs = np.array(resp.json()["embeddings"], dtype=np.float32)
    return vecs[0]

def search(query: str, top_k: int = TOP_K_DEFAULT) -> List[Dict[str, Any]]:
    client = MilvusClient(uri=DB_PATH)
    qv = embed_query(query).tolist()
    results = client.search(
        collection_name=COLLECTION,
        data=[qv],
        anns_field=VECTOR_FIELD,
        limit=top_k,
        output_fields=["id", "text", "doc_name", "chunk_id"]  # если нет этих полей — Milvus просто вернёт то, что есть
    )
    return results[0]

def search_grouped_by_doc(query: str, top_docs: int = 5, chunks_per_doc: int = 3, oversample: int = 80) -> List[Dict[str, Any]]:
    raw_hits = search(query, top_k=max(oversample, top_docs * chunks_per_doc * 2))
    by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for h in raw_hits:
        ent = h["entity"]
        by_doc[(ent.get("doc_name") or "unknown")].append({
            "text": ent.get("text",""),
            "score": h.get("score", 0.0),
            "chunk_id": ent.get("chunk_id"),
        })
    docs = []
    for doc_name, items in by_doc.items():
        items_sorted = sorted(items, key=lambda x: x["score"], reverse=True)[:chunks_per_doc]
        doc_score = max((x["score"] for x in items_sorted), default=0.0)
        docs.append({"doc_name": doc_name, "score": doc_score, "chunks": items_sorted})
    return sorted(docs, key=lambda d: d["score"], reverse=True)[:top_docs]