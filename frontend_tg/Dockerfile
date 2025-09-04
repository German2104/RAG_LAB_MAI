# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Базовые пакеты (логи, tzdata опционально)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tzdata curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Зависимости бота (aiogram) + то, чем он пользуется из backend (Milvus, парсеры и т.д.)
COPY requirements-bot.txt /app/requirements-bot.txt
RUN pip install --no-cache-dir -r /app/requirements-bot.txt

# Код: бот и backend-модули, т.к. бот импортирует backend.indexer
COPY frontend_tg /app/frontend_tg
COPY backend /app/backend

# Папка для загрузок (если том не смонтирован, всё равно будет существовать)
RUN mkdir -p /app/frontend_tg/uploads

# Скрипт запуска (ловит SIGTERM/SIGINT корректно)
COPY frontend_tg/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Переменные окружения (значения задашь при запуске/в compose)
ENV BOT_TOKEN=""
# Если твой embedding-клиент ожидает иной URL — передай EMBEDDING_SERVICE_URL окружением.
# ENV EMBEDDING_SERVICE_URL="http://host.docker.internal:8100/v1/models/jinaai-embeddings:predict"

# Экспонировать порты не требуется (бот сам инициирует соединение к Telegram)
ENTRYPOINT ["/app/start.sh"]