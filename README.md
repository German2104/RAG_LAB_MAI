# 📚 RAG_LAB_MAI

Телеграм-бот для **извлечения знаний из документов** с помощью Retrieval-Augmented Generation (RAG).  
Проект позволяет загружать файлы (PDF, DOCX, TXT), индексировать их в Milvus, а затем задавать вопросы и получать ответы через **GigaChat**.

---

## ✨ Возможности

- 📤 Загрузка документов (PDF, DOCX, TXT) прямо в Telegram.
- 🔎 Индексация текста в **Milvus Lite** с эмбеддингами `jinaai/jina-embeddings-v3`.
- 🧩 Поиск релевантных фрагментов (чанков) с группировкой по документам.
- 🤖 Генерация ответов через **GigaChat API** (OpenAI-совместимое API).
- ⚡ Поддержка GPU (`cuda`) / MPS (`Apple Silicon`) / CPU.
- 📖 Простая интеграция в локальную или облачную инфраструктуру.

---

## 🛠️ Технологии

- **Python 3.11+**
- [aiogram](https://docs.aiogram.dev/) — Telegram-бот
- [FastAPI](https://fastapi.tiangolo.com/) — сервис эмбеддингов
- [transformers](https://huggingface.co/docs/transformers/) — Jina Embeddings
- [pymilvus](https://milvus.io/) — векторная база
- [langchain-gigachat](https://github.com/sberdevices/langchain-gigachat) — работа с GigaChat
- [uvicorn](https://www.uvicorn.org/) — сервер API
- [pymupdf](https://pymupdf.readthedocs.io/) + [python-docx](https://python-docx.readthedocs.io/) — извлечение текста

---

## 📂 Архитектура проекта
```
RAG_LAB_MAI/
├── backend/                 # Бэкенд (индексация, поиск, RAG)
│   ├── config.py            # Конфигурация
│   ├── deploy.py            # FastAPI сервис эмбеддингов
│   ├── indexer.py           # Индексация документов
│   ├── searcher.py          # Поиск в Milvus
│   ├── rag_qa.py            # Логика RAG (QA через GigaChat)
│   └── gigachat_langchain.py# Обёртка для работы с GigaChat
│
├── frontend_tg/             # Телеграм-бот (aiogram)
│   └── app.py
│
├── db/                      # Milvus Lite база (игнорируется в git)
├── uploads/                 # Загруженные документы (игнорируются)
├── .env                     # Переменные окружения (не коммитится)
├── .gitignore
├── Dockerfile               # Контейнеризация
├── requirements.txt         # Зависимости (для dev/prod)
└── README.md
```
---

## 🚀 Запуск

### 1. Клонируй репозиторий и установи зависимости

```bash
git clone https://github.com/<your_repo>/RAG_LAB_MAI.git
cd RAG_LAB_MAI
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### 2. Создай .env
```
# --- Телеграм ---
BOT_TOKEN=1234567890:ABCDEF...   # токен бота от BotFather

# --- GigaChat OAuth ---
GIGACHAT_AUTH_URL=https://ngw.devices.sberbank.ru:9443/api/v2/oauth
GIGACHAT_CLIENT_ID=<ваш Client ID>
GIGACHAT_CLIENT_SECRET=<ваш Client Secret>
GIGACHAT_SCOPE=GIGACHAT_API_PERS
GIGACHAT_API_URL=https://gigachat.devices.sberbank.ru/api/v1
GIGACHAT_VERIFY_SSL=false

# --- Embeddings ---
EMBEDDING_MODEL_NAME=jinaai/jina-embeddings-v3
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8000

# --- Milvus ---
MILVUS_DB_PATH=./db/milvus.db
COLLECTION_NAME=pdf_embeddings
VECTOR_FIELD=vector
DIMENSION=1024
SEARCH_METRIC=IP
TOP_K_DEFAULT=10
```
3. Запусти сервис эмбеддингов
```
cd backend
python deploy.py

Сервис поднимется на http://localhost:8000.
```
4. Запусти бота
```
cd frontend_tg
python app.py

Теперь бот доступен в Telegram.
```

⸻

📖 Использование

	1.	В Telegram отправь /start.
	2.	Нажми «📤 Загрузить документ» и прикрепи файл (PDF, DOCX, TXT).
	3.	Подожди окончания индексации (бот уведомит).
	4.	Задай вопрос текстом → получи ответ из документов.

⸻

🔒 Безопасность

	•	🔑 Секреты (BOT_TOKEN, GIGACHAT_CLIENT_SECRET, токены) хранятся только в .env.
	•	🛡️ .gitignore защищает от случайного пуша чувствительных файлов:
	•	__pycache__/
	•	.venv/
	•	db/
	•	uploads/
	•	.env

⸻

📝 TODO

	•	Поддержка других форматов (Excel, HTML).
	•	Стриминг ответов из GigaChat.
	•	Веб-интерфейс (FastAPI + Gradio).
	•	Автоочистка старых индексов.

⸻

👨‍💻 Для разработчиков

	•	Линтинг: flake8, black
	•	Тесты: pytest
	•	Типизация: mypy

⸻

📜 Лицензия

MIT License © 2025

