# syntax=docker/dockerfile:1

FROM python:3.11-slim

# ========== базовые настройки ==========
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Установка зависимостей для pymupdf4llm, python-docx, numpy и т.п.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpoppler-cpp-dev \
        pkg-config \
        git \
        && rm -rf /var/lib/apt/lists/*

# ========== рабочая директория ==========
WORKDIR /app

# ========== зависимости ==========
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ========== исходный код ==========
COPY backend /app/backend

# по умолчанию будем использовать rag_answer как entrypoint
ENTRYPOINT ["python", "-m", "backend.rag_answer"]