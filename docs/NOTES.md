# Заметки

## Окружение

- Mac: MacBook Air M4, 16 ГБ, macOS 26.6.2. Пассивное охлаждение → долгие локальные нагрузки будут троттлить.
- Есть: `uv` (~/.local/bin/uv), `node v24.19.0`, `git`, системный `python3 3.9.6` (не использовать — проект на 3.12 через uv).
- Нет: `brew`, `docker`, `ollama`. 7z распаковываем через `uvx --from py7zr py7zr x <file> <dir>`.
- zsh: `echo ===` ломается (equals-expansion) — в командах использовать `echo "---"`.

## Структура репо

```
src/mechanic/
  vehicles.py            # 5 машин MVP (+ ключи источников данных)
  server.py              # FastAPI: /torque, /api/catalog, /api/sim/{device}, /api/sensors/{device}
  torque/pids.py         # PID-ы, ключи Torque (k5, kc, ...)
  torque/receiver.py     # парсер Torque web upload → store, ответ "OK!"
  torque/store.py        # SQLite: readings / sensor_meta / dtcs; latest(), trend()
  torque/simulator.py    # VehicleModel (физика-лайт + 6 неисправностей), TorqueSimulator (HTTP), CLI
  data/common.py         # Doc-схема, PoliteFetcher (кэш + 1 req/s), looks_english
  data/stackexchange.py  # дамп SE → jsonl
  data/carcarekiosk.py   # скрапер how-to
  data/startmycar.py     # скрапер отчётов владельцев
tests/test_torque.py
data/raw/ (дамп, gitignored)  data/cache/ (http-кэш, sqlite, логи; gitignored)  data/processed/ (jsonl; gitignored)
```

## Команды

```bash
uv run pytest -q                                   # тесты
uv run ruff check --fix src tests && uv run ruff format src tests
uv run uvicorn mechanic.server:app --port 8000 --reload   # или preview_start backend
uv run python -m mechanic.torque.simulator --vehicle audi_a4_b8 --mode idle --fault vacuum_leak --time-scale 20
uv run python -m mechanic.data.stackexchange       # ~3 с
uv run python -m mechanic.data.carcarekiosk        # ~5 мин с нуля, из кэша — секунды
uv run python -m mechanic.data.startmycar          # долго (сотни запросов по 1/с), из кэша — быстро
# запустить симулятор через API:
curl -X POST localhost:8000/api/sim/demo -H 'content-type: application/json' -d '{"vehicle":"audi_a4_b8","mode":"idle","fault":"vacuum_leak","time_scale":20}'
curl localhost:8000/api/sensors/demo
```

## Источники данных (замеры 2026-09-17)

| Источник | Что | Объём | Доступ |
|---|---|---|---|
| mechanics.stackexchange.com | дамп `archive.org/download/stackexchange_20251231/stackexchange_20251231/mechanics.stackexchange.com.7z` (73 МБ) | 28 380 вопросов, 40 675 ответов → 20 996 тредов с ответом score≥1 или accepted, ~39 МБ текста | CC BY-SA; robots.txt сайта `Disallow: /` + `ai-train=no` → только дамп |
| carcarekiosk.com | `/videos/<Make>/<Model>/<genYear>` → ~40 (F-150: 81) страниц `/video/...` на поколение; текст = только «Video Description» (шаги в видео) | ~240 страниц на 5 машин | robots.txt разрешает; sitemap.xml есть |
| startmycar.com | `/us/<make>/<model>/problems[/pageN]`, 30 карточек/стр.; карточка: `div.js-report[data-denunciaid]`, `h3 a`, `.text-ellipsis` (год/комплектация/пробег), `.TagLink`, `.js-report-body`; решено = класс `solucionado`; ответы на детальной стр.: `.CommentCard` (`--solution` = лучший), `.CommentCard__text` | отчёты: accord 2282, f-150 2025, corolla 1000, civic 708, a4 338 (все поколения, много испанского) | robots.txt разрешает; sitemap нет |

carcarekiosk (готово 2026-09-17): 238 документов (a4 36, accord 41, f150 81, civic 41, corolla 39), в среднем ~1.3k символов, 243 запроса, ~15 мин (сайт отвечает ~3–4 с на страницу).

SE: матчинг машин по тегу модели (+ год из заголовка, если есть). Результат vehicle_ids: a4_b8 17, accord_9 185, f150_13 51, civic_10 290, corolla_11 108. Марка определена у ~3.7k тредов, остальные — общие вопросы.

## Грабли и неочевидное

- Браузер даёт микрофон только по HTTPS (или localhost).
- Cloudflare Tunnel не проксирует UDP → WebRTC не пройдёт, используем WebSocket.
- Каждый процесс с CUDA съедает ~0.3–0.5 ГБ VRAM только на контекст (?) — проверить на Vast; держать TTS в одном процессе.
- Qwen3-30B-A3B в Q4 весит ~17–18 ГБ (?) — в 16 ГБ только ~Q3 или Q4 с экспертами на CPU (`--n-cpu-moe` в llama.cpp).
- Torque web upload: подтверждено — GET, ключи `k<hex pid>` без ведущих нулей (`k5`, `kc`, `kff1005`), метаданные `userFullName<pid>` и т.п., ответ `OK!`. В HA-интеграции единица приходит как `\xC2\xB0C` — заменяем на `°`.
- В симуляторе `dt = interval_s * time_scale`; при тестах с маленьким interval нужен большой time_scale, иначе неисправность не успевает «развиться».
- carcarekiosk: страница поколения, напр. `Audi/A4_Quattro/2009`, покрывает все годы поколения («produced from 2009 - 2016»).

## Бенчмарки

_(таблицы LLM / TTS / end-to-end задержки — в день 3 и 6)_

## Материал для README

- Зачем такой стек: см. DECISIONS.md.
- Как замеряется задержка: client-side VAD (конец речи) → первый аудиосэмпл ответа в браузере; плюс серверная разбивка по этапам; benchmark-режим с фиксированными WAV → p50/p95.
- Реальный Torque Pro подключается без изменений кода: Torque → Settings → Data Logging & Upload → Webserver URL → `https://<backend>/torque`.
- Симулятор говорит с бэкендом по тому же HTTP-протоколу, что и телефон.
- Атрибуция: Stack Exchange контент — CC BY-SA, ссылка на каждый тред в ответах/карточках.
