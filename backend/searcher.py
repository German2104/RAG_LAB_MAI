# backend/searcher.py
from __future__ import annotations
import json
from argparse import ArgumentParser
from typing import List, Tuple

import numpy as np
from pymilvus import MilvusClient
from .embedding_client import emb_text

# ---- Конфиг Milvus (совпадает с indexer.py) ----
DB_PATH      = "./milvus.db"
COLLECTION   = "pdf_embeddings"
VECTOR_FIELD = "vector"
TEXT_FIELD   = "text"
DIMENSION    = 1024

def search(query: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """Ищет top_k ближайших текстов в Milvus и возвращает (text, distance)."""
    client = MilvusClient(uri=DB_PATH)
    if not client.has_collection(COLLECTION):
        raise RuntimeError(f"Коллекция {COLLECTION} отсутствует. Сначала проиндексируй документы.")

    vec = emb_text(query)
    res = client.search(
        collection_name=COLLECTION,
        data=[vec],
        limit=top_k,
        search_params={"metric_type": "IP", "params": {}},
        output_fields=[TEXT_FIELD],
    )
    # res — список на запрос; берём первый
    first = res[0] if res else []
    pairs: List[Tuple[str, float]] = []
    for item in first:
        text = item["entity"][TEXT_FIELD]
        distance = float(item["distance"])
        pairs.append((text, distance))
    return pairs

def create_parser() -> ArgumentParser:
    p = ArgumentParser(description="Поиск по Milvus (pdf_embeddings) с Jina Embeddings v3")
    p.add_argument("--query", "-q", type=str, help="Текст запроса. Если не задан, читается из stdin.")
    p.add_argument("--top-k", type=int, default=5, help="Количество результатов (default 5).")
    return p

def main():
    parser = create_parser()
    ns = parser.parse_args()

    if ns.query is None:
        # читаем весь stdin
        try:
            ns.query = input().strip()
        except EOFError:
            ns.query = ""

    if not ns.query:
        raise SystemExit("Запрос пуст. Передай --query или введи строку через stdin.")

    results = search(ns.query, ns.top_k)
    print(json.dumps(results, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()