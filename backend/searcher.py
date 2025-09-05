#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any
import logging

from pymilvus import MilvusClient
from pymilvus.exceptions import MilvusException
from backend.config import DB_PATH, COLLECTION, VECTOR_FIELD, DIMENSION, SERVICE_URL, TOP_K_DEFAULT

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

def _ensure_loaded(client: MilvusClient):
    try:
        client.load_collection(collection_name=COLLECTION)
    except Exception:
        pass

def _ensure_index(client: MilvusClient):
    """
    Создаём индекс, если его ещё нет. Используем именно IndexParams, а не dict.
    """
    try:
        # если индекса нет — create_index не упадёт при повторном создании в AUTOINDEX
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name=VECTOR_FIELD,
            index_type="AUTOINDEX",
            metric_type="IP",
            params={},  # AUTOINDEX не требует параметров
        )
        client.create_index(collection_name=COLLECTION, index_params=index_params)
        _ensure_loaded(client)
    except Exception as e:
        logger.warning("Не удалось создать индекс (возможно уже есть): %s", e)

def embed_query(query: str) -> np.ndarray:
    resp = requests.post(f"{SERVICE_URL}/embed", json={"texts": [query]}, timeout=60)
    resp.raise_for_status()
    vecs = np.array(resp.json()["embeddings"], dtype=np.float32)
    return vecs[0]

def search(query: str, top_k: int = TOP_K_DEFAULT) -> List[Dict[str, Any]]:
    client = MilvusClient(uri=DB_PATH)
    _ensure_loaded(client)
    qv = embed_query(query).tolist()

    try:
        results = client.search(
            collection_name=COLLECTION,
            data=[qv],
            anns_field=VECTOR_FIELD,
            limit=top_k,
            output_fields=["id", "text", "doc_name", "chunk_id"],
        )
    except MilvusException as e:
        # код 700 = index not found
        if "index not found" in str(e).lower():
            logger.info("Индекс отсутствует — создаю и повторяю поиск…")
            _ensure_index(client)
            results = client.search(
                collection_name=COLLECTION,
                data=[qv],
                anns_field=VECTOR_FIELD,
                limit=top_k,
                output_fields=["id", "text", "doc_name", "chunk_id"],
            )
        else:
            raise

    hits = results[0]
    logger.info("Поиск '%s' -> %d хитов", query, len(hits))
    for i, h in enumerate(hits[:10]):
        ent = h.get("entity", {})
        logger.info("#%d doc=%s chunk=%s score=%.4f",
                    i + 1, ent.get("doc_name", "unknown"),
                    ent.get("chunk_id", "-"),
                    h.get("score", 0.0))
    return hits

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
    docs_sorted = sorted(docs, key=lambda d: d["score"], reverse=True)[:top_docs]

    logger.info("=== Топ-%d документов '%s' ===", top_docs, query)
    for d in docs_sorted:
        logger.info("Документ %s (score=%.4f, чанков=%d)", d["doc_name"], d["score"], len(d["chunks"]))
    return docs_sorted