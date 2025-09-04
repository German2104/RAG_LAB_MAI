#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${BOT_TOKEN:-}" ]]; then
  echo "ERROR: BOT_TOKEN не задан" >&2
  exit 1
fi

# создадим uploads на всякий случай (если не смонтирован том)
mkdir -p /app/frontend_tg/uploads

# Запуск бота
exec python -m frontend_tg.app