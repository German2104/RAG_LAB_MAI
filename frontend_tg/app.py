import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import asyncio
import logging
import sys
from os import getenv
from pathlib import Path
from datetime import datetime

from aiogram import Bot, Dispatcher, html, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import (
    Message,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    CallbackQuery,
    ReplyKeyboardRemove,
)
from backend.indexer import index_file
from backend.searcher import search  # подключаем поиск
from backend.rag_qa import answer_with_top_docs, answer_with_top_chunks

# Bot token can be obtained via https://t.me/BotFather
TOKEN = getenv("BOT_TOKEN")
print(TOKEN)
assert not (TOKEN is None), "Экспортируйте токен"

# === Клавиатуры ===
start_keyboard = InlineKeyboardMarkup(
    inline_keyboard=[
        [InlineKeyboardButton(text="📤 Загрузить документ", callback_data="upload")],
        [InlineKeyboardButton(text="ℹ️ Помощь", callback_data="help")],
    ]
)

# All handlers should be attached to the Router (or Dispatcher)

dp = Dispatcher()

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    Этот хендлер получает сообщения с командой /start
    """
    await message.answer(
        f"Hello, {html.bold(message.from_user.full_name)}!\n"
        f"Этот бот поможет тебе изучить документ быстрее.\n\n"
        f"Нажми «📤 Загрузить документ», затем прикрепи файл как документ.",
        reply_markup=start_keyboard,
    )


@dp.callback_query(F.data == "help")
async def on_help(callback: CallbackQuery) -> None:
    """
    Подсказка по использованию бота
    """
    await callback.message.answer(
        "Как загрузить документ:\n"
        "1) Нажми «📤 Загрузить документ»\n"
        "2) Нажми скрепку (📎) → «Файл»/«Документ»\n"
        "3) Выбери файл (PDF/DOCX/TXT и т.п.)\n\n"
        "После загрузки бот сохранит файл на сервере в папку ./uploads/",
        reply_markup=ReplyKeyboardRemove(),
    )
    await callback.answer()  # закрыть "часики" у кнопки


@dp.callback_query(F.data == "upload")
async def on_upload_click(callback: CallbackQuery) -> None:
    """
    По нажатию на кнопку «Загрузить документ» просим прислать файл
    """
    await callback.message.answer(
        "Пришли документ как файл (не фото). Можно несколько — по одному.\n"
        "Для отмены просто ничего не отправляй или введи /start заново."
    )
    await callback.answer()

@dp.message(F.document)
async def handle_document(message: Message, bot: Bot) -> None:
    """
    Принимаем документ и сохраняем его в ./uploads
    """
    upload_dir = Path("./uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    doc = message.document
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    original_name = doc.file_name or "file"
    safe_name = f"{timestamp}__{original_name}"
    dest_path = upload_dir / safe_name

    # Скачиваем файл
    await bot.download(doc, destination=dest_path)

    await message.answer(
        "✅ Файл получен.\n"
        f"Путь: <code>{dest_path.as_posix()}</code>\n"
        "Начинаю индексацию… Это может занять немного времени."
    )

    def _run():
        index_file(str(dest_path))

    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, _run)
        # После успешной индексации показываем клавиатуру
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="💬 Начать ответы по документам", callback_data="start_qa")]
            ]
        )
        await message.answer("✅ Индексация завершена. Файл добавлен в БД Milvus.", reply_markup=keyboard)
    except Exception as e:
        await message.answer(f"❌ Ошибка индексации: <code>{e}</code>")

@dp.callback_query(F.data == "start_qa")
async def on_start_qa(callback: CallbackQuery) -> None:
    await callback.message.answer(
        "📖 Отлично! Теперь пришли мне любой вопрос по документам, и я найду для тебя ответы.",
        reply_markup=ReplyKeyboardRemove()
    )
    await callback.answer()




@dp.message(F.text)
async def handle_question(message: Message) -> None:
    query = message.text.strip()
    if not query:
        return

    await message.answer("🔎 Ищу ответ по документам…")

    try:
        # Вариант A: если в индексе есть doc_name — используем документы
        # answer = answer_with_top_docs(query, top_docs=10, chunks_per_doc=3)

        # Вариант B: если храним только чанки (без doc_name) — используем топ чанков
        answer = answer_with_top_chunks(query, top_k=10)

        await message.answer(f"💡 Ответ:\n\n{answer}")
    except Exception as e:
        await message.answer(f"⚠️ Ошибка поиска/генерации: <code>{e}</code>")

        
async def main() -> None:
    # Initialize Bot instance with default bot properties which will be passed to all API calls
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    # And the run events dispatching
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())