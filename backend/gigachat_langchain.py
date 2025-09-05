#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from functools import lru_cache
from typing import Optional
from pathlib import Path

# auto-load .env из корня проекта
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    # сначала корень репо, затем рядом с файлом (на всякий)
    for p in (Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"):
        if p.exists():
            load_dotenv(p)  # не шумим, просто загружаем
except Exception:
    pass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat

def _bool_env(name: str, default: bool = True) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

@lru_cache(maxsize=1)
def get_gigachat() -> GigaChat:
    """
    credentials:
      - если задан GIGACHAT_API_KEY -> считаем это готовым Bearer
      - иначе берём GIGACHAT_BASIC (Authorization key base64) или CLIENT_SECRET_B64
      - на всякий случай поддерживаем твою старую переменную GIGACHAT_CLIENT_SECRET,
        если нет CLIENT_ID (т.е. строка похоже уже base64 Authorization key)
    """
    cred = (
        os.getenv("GIGACHAT_API_KEY")
        or os.getenv("GIGACHAT_BASIC")
        or os.getenv("GIGACHAT_CLIENT_SECRET_B64")
    )
    if not cred:
        # если нет CLIENT_ID и есть CLIENT_SECRET — трактуем его как base64 Authorization key
        if not os.getenv("GIGACHAT_CLIENT_ID") and os.getenv("GIGACHAT_CLIENT_SECRET"):
            cred = os.getenv("GIGACHAT_CLIENT_SECRET")

    if not cred:
        raise RuntimeError(
            "Не найден ни GIGACHAT_API_KEY (Bearer), ни GIGACHAT_BASIC (Authorization key base64). "
            "Проверь .env."
        )

    model = os.getenv("GIGACHAT_MODEL", "GigaChat-2")
    scope = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
    verify_ssl = _bool_env("GIGACHAT_VERIFY_SSL", True)

    return GigaChat(
        credentials=cred,
        model=model,
        scope=scope,
        verify_ssl_certs=verify_ssl,
        timeout=120,
        profanity_check=False,
        streaming=False,
    )

def lc_answer(system_prompt: str, user_prompt: str) -> str:
    giga = get_gigachat()
    msgs = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    res = giga.invoke(msgs)
    return res.content