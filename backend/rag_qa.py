#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from textwrap import shorten
from typing import List, Dict, Any

from backend.searcher import search, search_grouped_by_doc
from backend.gigachat_langchain import lc_answer  # LangChain-клиент GigaChat

# --------- сборка контекста ---------

def build_context_from_chunks(hits: List[Dict[str, Any]], max_chars: int = 8000) -> str:
    parts: List[str] = []
    for i, h in enumerate(hits, start=1):
        score = h.get("score", 0.0)
        ent = h.get("entity", {})
        text = (ent.get("text") or "").replace("\n", " ").strip()
        doc_name = ent.get("doc_name") or ""
        prefix = f"[{i} score={score:.6f}] "
        if doc_name:
            prefix += f"{doc_name}: "
        parts.append(prefix + shorten(text, width=900, placeholder="…"))
    ctx = "\n".join(parts)
    return ctx[:max_chars] + ("…" if len(ctx) > max_chars else "")

def build_context_from_docs(docs: List[Dict[str, Any]], max_chars: int = 8000) -> str:
    rows: List[str] = []
    for d_i, d in enumerate(docs, start=1):
        header = f"[DOC {d_i}] {d['doc_name']} (score={d['score']:.6f})"
        rows.append(header)
        for c in d["chunks"]:
            rows.append("• " + shorten((c["text"] or "").replace("\n", " ").strip(), width=900, placeholder="…"))
        rows.append("")
    ctx = "\n".join(rows)
    return ctx[:max_chars] + ("…" if len(ctx) > max_chars else "")

# --------- вызов GigaChat через LangChain ---------

def gigachat_answer(system_prompt: str, user_prompt: str) -> str:
    """Синоним для обратной совместимости со старыми вызовами."""
    return lc_answer(system_prompt, user_prompt)

# --------- публичные функции ---------

def answer_with_top_chunks(query: str, top_k: int = 10) -> str:
    hits = search(query, top_k=top_k)
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