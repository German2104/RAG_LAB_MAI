#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import requests
from textwrap import shorten

from typing import List, Dict, Any

# наши модули/константы
from config import TOP_K_DEFAULT, SERVICE_URL
from backend.searcher import search, search_grouped_by_doc  # поиск по чанкам/докам

# ---------- Сбор контекста ----------

def build_context_from_chunks(hits: List[Dict[str, Any]], max_chars: int = 8000) -> str:
    """
    hits: элементы, как возвращает MilvusClient.search()[0]:
      {"score": float, "entity": {"text": str, "doc_name"?: str, "chunk_id"?: int, ...}}
    Склеиваем в компактный контекст.
    """
    parts: List[str] = []
    for i, h in enumerate(hits, start=1):
        score = h.get("score", 0.0)
        ent = h.get("entity", {})
        text = (ent.get("text") or "").replace("\n", " ").strip()
        doc_name = ent.get("doc_name") or ""
        prefix = f"[{i} score={score:.3f}] "
        if doc_name:
            prefix += f"{doc_name}: "
        parts.append(prefix + shorten(text, width=900, placeholder="…"))
    ctx = "\n".join(parts)
    return ctx[:max_chars] + ("…" if len(ctx) > max_chars else "")


def build_context_from_docs(docs: List[Dict[str, Any]], max_chars: int = 8000) -> str:
    """
    docs: результат search_grouped_by_doc(...)
      [{"doc_name": str, "score": float, "chunks": [{"text": str, "score": float, "chunk_id": int}, ...]}, ...]
    """
    rows: List[str] = []
    for d_i, d in enumerate(docs, start=1):
        header = f"[DOC {d_i}] {d['doc_name']} (score={d['score']:.3f})"
        rows.append(header)
        for c in d["chunks"]:
            rows.append("• " + shorten((c["text"] or "").replace("\n"," ").strip(), width=900, placeholder="…"))
        rows.append("")  # разделитель
    ctx = "\n".join(rows)
    return ctx[:max_chars] + ("…" if len(ctx) > max_chars else "")

# ---------- Вызов GigaChat (OpenAI-совместимое API) ----------

def gigachat_answer(system_prompt: str, user_prompt: str) -> str:
    """
    Вызывает GigaChat через OpenAI-совместимое API.
    Требуются переменные окружения:
      GIGACHAT_API_URL  (например: https://gigachat.devices.sberbank.ru/api/v1)
      GIGACHAT_API_KEY  (Bearer-токен)
    """
    base_url = os.getenv("GIGACHAT_API_URL")
    api_key  = os.getenv("GIGACHAT_API_KEY")

    assert base_url, "Укажи GIGACHAT_API_URL в переменных окружения"
    assert api_key,  "Укажи GIGACHAT_API_KEY в переменных окружения"

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "GigaChat",             # если у тебя другой ID модели — поставь его
        "temperature": 0.2,
        "max_tokens": 800,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    # стандартный формат OpenAI-compatible
    return data["choices"][0]["message"]["content"]

# ---------- Публичные функции ----------

def answer_with_top_chunks(query: str, top_k: int = 10) -> str:
    """
    Ищем top_k похожих чанков и отвечаем через GigaChat.
    Подходит, если в схеме нет doc_name.
    """
    hits = search(query, top_k=top_k)  # backend.searcher.search
    if not hits:
        return "Не нашёл релевантного контента в базе. Попробуй переформулировать вопрос."

    context = build_context_from_chunks(hits, max_chars=8000)
    system_prompt = (
        "Ты помощник, отвечающий строго по предоставленному контексту. "
        "Если ответа нет в контексте — честно скажи об этом. Отвечай кратко и по делу."
    )
    user_prompt = f"Вопрос: {query}\n\nКонтекст:\n{context}\n\nДай связанный ответ на русском языке."
    return gigachat_answer(system_prompt, user_prompt)


def answer_with_top_docs(query: str, top_docs: int = 5, chunks_per_doc: int = 3) -> str:
    """
    Ищем top_docs документов (по нескольким лучшим чанкам на документ), затем отвечаем через GigaChat.
    Работает, если в коллекции есть поле doc_name (иначе документы будут «unknown»).
    """
    docs = search_grouped_by_doc(query, top_docs=top_docs, chunks_per_doc=chunks_per_doc, oversample=80)
    if not docs:
        return "Не нашёл релевантного контента в базе. Попробуй переформулировать вопрос."

    context = build_context_from_docs(docs, max_chars=8000)
    system_prompt = (
        "Ты помощник, отвечающий строго по предоставленному контексту. "
        "Если ответа нет в контексте — честно скажи об этом. Отвечай кратко и по делу."
    )
    user_prompt = (
        f"Вопрос: {query}\n\n"
        f"Контекст: ниже собраны фрагменты из топ-{len(docs)} документов.\n"
        f"{context}\n\nСформулируй ответ на русском."
    )
    return gigachat_answer(system_prompt, user_prompt)