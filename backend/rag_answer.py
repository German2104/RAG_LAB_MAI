# backend/rag_answer.py
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

# ---- Конфиг генеративной модели ----
import openai
import os

# Настройка OpenAI API (добавьте свой API ключ в переменную окружения)
openai.api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")  # Заглушка ссылки на API ключ

def generate_answer(context_chunks: List[str], question: str) -> str:
    """Генерация ответа через OpenAI API."""
    try:
        context = "\n\n".join(f"{i+1}. {c}" for i, c in enumerate(context_chunks))
        system_prompt = (
            "Ты — краткий и точный помощник. Отвечай по существу, используя ТОЛЬКО факты из контекста. "
            "Если ответа нет в контексте — скажи, что в контексте информации недостаточно."
        )
        user_prompt = f"КОНТЕКСТ:\n{context}\n\nВОПРОС: {question}"
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Заглушка модели - замените на нужную
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=512,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        # Заглушка для обработки ошибок
        return f"Ошибка при генерации ответа: {str(e)}\n\nКонтекст для ответа:\n{context}\n\nВопрос: {question}"

def milvus_search(query: str, top_k: int = 5) -> List[Tuple[str, float]]:
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
    first = res[0] if res else []
    out: List[Tuple[str, float]] = []
    for item in first:
        text = item["entity"][TEXT_FIELD]
        dist = float(item["distance"])
        out.append((text, dist))
    return out

def create_parser() -> ArgumentParser:
    p = ArgumentParser(description="RAG: поиск в Milvus + ответ через OpenAI API")
    p.add_argument("--query", "-q", type=str, help="Вопрос пользователя. Если не задан, читается из stdin.")
    p.add_argument("--top-k", type=int, default=5, help="Сколько фрагментов брать из БД (default 5).")
    p.add_argument("--show-hits", action="store_true", help="Показать JSON с найденными фрагментами.")
    return p

def main():
    parser = create_parser()
    ns = parser.parse_args()

    if ns.query is None:
        try:
            ns.query = input().strip()
        except EOFError:
            ns.query = ""

    if not ns.query:
        raise SystemExit("Вопрос пуст. Передай --query или введи строку через stdin.")

    hits = milvus_search(ns.query, ns.top_k)
    if ns.show_hits:
        print(json.dumps(hits, indent=4, ensure_ascii=False))

    context_chunks = [t for (t, _d) in hits]
    answer = generate_answer(context_chunks, ns.query)
    print(answer)

if __name__ == "__main__":
    main()