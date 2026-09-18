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
  knowledge/dtc.py       # OBDex: 9533 кода; normalize_code() понимает "P zero one seven one"
  knowledge/search.py    # гибридный поиск: bm25s + bge-small (fastembed, CPU), RRF, буст по машине
  agent/tools.py         # 8 тулов + Session + ToolRunner
  agent/prompt.py        # системный промпт (короткий: это кэшируемый префикс)
  agent/loop.py          # стриминг по предложениям, фразы-заглушки, вызовы тулов, тайминги
  evals/run.py           # прогон evals/scenarios.yaml против любого OpenAI-совместимого сервера
  config.py              # load_env(): .env → os.environ (без python-dotenv)
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
uv run python -m mechanic.knowledge.dtc            # пересобрать кэш кодов
uv run python -m mechanic.knowledge.search build   # индекс (~56k пассажей; bm25 быстро, эмбеддинги долго)
uv run python -m mechanic.knowledge.search query "high fuel trim at idle" --vehicle audi_a4_b8
uv run python -m mechanic.evals.run --model <name> --base-url http://127.0.0.1:8001/v1
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
- Тесты фейкового LLM: `AsyncOpenAI`-клиент подменяется объектом со скриптом чанков; список `messages` мутируется циклом, поэтому в фейке его надо копировать.
- `pytest` не видит `tests` как пакет — общие хелперы кладём в `tests/conftest.py` и импортируем как `from conftest import ...`.
- Ruff: длинные строки описаний тулов и промпта разрешены через per-file-ignores (E501).
- **bm25s**: для запроса нужен `bm25s.tokenize(q, return_ids=False)` → список строк. Если передать `ids` от свежего токенизатора, они ссылаются на ДРУГОЙ словарь и выдача будет мусорной (попались 2026-09-17). Индексация — наоборот, через ids (`tokenize_corpus`). Тест `tests/test_search.py` это стережёт.
- Секреты: `.env.example` в git как документация, `.env` — в gitignore (`git check-ignore -v .env` проверяет). `load_env()` использует `os.environ.setdefault`, так что на арендованной машине переменные окружения контейнера перебивают файл. `idle_shutdown.py` берёт id инстанса из `CONTAINER_ID` (его ставит сам Vast внутри инстанса), а с ноутбука — из `VAST_INSTANCE_ID`.
- MacBook Air M4 без вентилятора: fastembed забирает ~490% CPU и ноутбук сильно греется + троттлит. Для локальных прогонов `--threads 4`; лучше считать эмбеддинги на арендованной GPU (`--dense-only`).

## Аренда GPU: грабли (2026-09-18)

- **`cpu_cores` в оффере Vast — ядра ХОСТА, не наши.** Наша доля — `cpu_cores_effective`
  (и оперативка тоже: `cpu_ram` — хостовая, наша ≈ `cpu_ram * gpu_frac`). Первый инстанс был
  выбран по `cpu_cores>=8` и оказался с 7 эффективными ядрами. Фильтровать и сортировать
  надо по `cpu_cores_effective`; в выдаче `--raw` это поле есть.
- **fastembed/onnxruntime плодит потоки по числу ВИДИМЫХ ядер.** В контейнере `nproc` показывает
  ядра хоста (56), а cgroup даёт 6.7 (`/sys/fs/cgroup/cpu.max` → `672000 100000`). Результат:
  98 потоков на 6.7 ядра, 8 минут без единого пассажа. Всегда передавать `--threads N` по квоте
  (`cpu.max` / 100000), плюс `OMP_NUM_THREADS`.
- **Готовые бинарники llama.cpp (b11037) требуют glibc 2.38+**, то есть Ubuntu 24.04.
  Образ `pytorch/pytorch` — Ubuntu 22.04 / glibc 2.35, бинарники не стартуют
  (`GLIBC_2.38 not found`), а nvcc и gcc в образе нет — собрать из исходников тоже нельзя без
  доустановки всего тулчейна. Рабочий образ: **`nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04`**.
- Для llama.cpp нужны ДВА архива релиза: `llama-<build>-bin-ubuntu-cuda-12.8-x64.tar.gz` и
  `cudart-<...>.tar.gz` (libcudart/libcublas), плюс `apt install libgomp1`. Всё распаковывается
  плоско в одну папку; `LD_LIBRARY_PATH` указывает на неё же.
- **`vastai create instance` без `--cancel-unavail` при занятой машине молча создаёт инстанс в
  состоянии `stopped`** и возвращает `success: False` с номером контракта. `start` потом отвечает
  «Required resources are currently unavailable, state change queued». Всегда передавать
  `--cancel-unavail`.
- `vastai destroy instance` спрашивает подтверждение — в скрипте `yes y | vastai destroy ...`.
- **`pkill -f <шаблон>` по ssh убивает собственную сессию**, если шаблон совпадает с её же
  командной строкой. Убивать по PID.
- У инстанса есть собственный ключ **`CONTAINER_API_KEY`** (в `/proc/1/environ`, там же
  `CONTAINER_ID`) — им контейнер останавливает сам себя. Ключ аккаунта на арендованную машину
  класть не надо.
- Переменные Vast (`CONTAINER_ID` и др.) видны в окружении PID 1, но НЕ в ssh-сессии —
  читать через `tr '\0' '\n' < /proc/1/environ`.

## Модели: проверенные имена и размеры (2026-09-18)

| Репозиторий | Файл | Размер |
|---|---|---|
| `ggml-org/gpt-oss-20b-GGUF` | `gpt-oss-20b-MXFP4.gguf` | 11.3 ГБ |
| `unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF` | `Qwen3-30B-A3B-Instruct-2507-UD-Q3_K_XL.gguf` | 12.9 ГБ |
| `unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF` | `Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf` | 16.5 ГБ |
| `Qwen/Qwen3-14B-AWQ` | (vLLM, не GGUF) | — |

## Качество поиска (BM25-only, 2026-09-17)

Проверка на 3 запросах, поиск ~7 мс:
- «high fuel trim at idle but normal on the highway» → тред SE «P0171: what to look at next…» (верно).
- «engine overheating at idle but fine while driving» → треды про перегрев Civic/Mondeo (верно).
- «how do I check the coolant level» + vehicle=audi_a4_b8 → страница carcarekiosk именно про A4 (буст по машине работает).

Плотные векторы ещё не посчитаны: `data/index/dense.npy` отсутствует, `SearchIndex.has_dense == False`, поиск работает только на BM25.

## Бенчмарки

### LLM: VRAM (2026-09-18, RTX A4000 16376 MiB, llama.cpp b11037 CUDA 12.8)

Пик замерен семплером `nvidia-smi` раз в 0.5 с во время загрузки и генерации. Базовая линия 16 MiB.

| Кандидат | Контекст | Пик VRAM | Остаётся на TTS | Загрузка |
|---|---|---|---|---|
| gpt-oss-20b MXFP4 | 16k | **11 688 MiB** | ~4.6 ГБ | 16.0 с |
| Qwen3-30B-A3B UD-Q3_K_XL | 16k | **14 886 MiB** | ~1.4 ГБ | 18.3 с |
| Qwen3-30B-A3B UD-Q3_K_XL | 8k | **14 110 MiB** | ~2.2 ГБ | 11.5 с |
| Qwen3-30B-A3B UD-Q4_K_XL + `--n-cpu-moe` | — | не замерено | — | — |
| Qwen3-14B AWQ (vLLM) | — | не замерено | — | — |

Вывод по памяти: урезание контекста с 16k до 8k экономит всего **776 MiB** — KV-кэш у A3B маленький
(активны 3B из 30B), основное занимают веса. То есть «уменьшить контекст» нас не спасает,
и выбор между gpt-oss-20b и Qwen3-30B — это выбор между «хватает места на выразительный TTS»
и «умнее, но TTS придётся брать крошечный».

### LLM: скорость — ПРЕДВАРИТЕЛЬНО, цифры загрязнены

Замеры сделаны, пока на тех же ядрах считался dense-индекс (llama.cpp использует CPU для
семплинга), поэтому tok/s занижены и их надо переснять на свободной машине.

| Кандидат | TTFT | tok/s | Комментарий |
|---|---|---|---|
| gpt-oss-20b MXFP4 16k | 4961 мс (?) | 110.4 (?) | **TTFT 5 с** — модель сначала выдаёт reasoning-токены; для голоса это приговор, если не снизить reasoning effort |
| Qwen3-30B-A3B Q3 16k | 559 мс (?) | 87.5 (?) | |
| Qwen3-30B-A3B Q3 8k | 287 мс (?) | 42.3 (?) | tok/s сильно скачет от загрузки CPU — переснять |

### gpt-oss-20b: reasoning effort решает судьбу кандидата (2026-09-18)

llama-server принимает `chat_template_kwargs: {"reasoning_effort": "..."}` прямо в теле запроса.
Меряли время до первого **content**-токена (reasoning уходит в `delta.reasoning_content`, вслух
его не произносят), `max_tokens: 250`:

| reasoning_effort | до первого слова | до первого токена | reasoning-токенов | ответ |
|---|---|---|---|---|
| **low** | **1195 мс** | 781 мс | 7 | нормальный |
| medium | 3774 мс | 118 мс | 223 | нормальный |
| high | — | 484 мс | 247 | **ответа нет**: весь бюджет токенов съеден размышлениями |
| default (не задан) | 3213 мс | 573 мс | 178 | нормальный |

Вывод: без `reasoning_effort: low` gpt-oss-20b для голоса непригоден, с ним — 1.2 с до первого
слова. Это надо зашить в конфиг LLM-клиента, а не оставлять на умолчание. Отдельно: при `high`
модель вообще не доходит до ответа в разумном бюджете — если бы мерили только «умность» по
тексту ответа, кандидат выглядел бы сломанным.

### LLM: итоговый прогон (2026-09-18, 32 сценария, A4000 16376 MiB)

После правки «машина в префиксе» (см. коммит `018f2b9`). Контекст 16k у всех.

| Кандидат | Пик VRAM | На TTS | tok/s | Сценарии | Тулы | Содержание | p50 1-й фразы | p95 |
|---|---|---|---|---|---|---|---|---|
| gpt-oss-20b MXFP4 (reasoning low) | 11 700 | 4.6 ГБ | 116.6 | 65.6% | 84.8% | 75.8% | 881 мс | 1650 мс |
| Qwen3-30B-A3B **Q2_K_XL** | 12 960 | 3.3 ГБ | 120.6 | 71.9% | 84.8% | 75.8% | 630 мс | 1749 мс |
| Qwen3-30B-A3B Q3_K_XL | 14 908 | 1.4 ГБ | 106.3 | 71.9% | 78.8% | 81.8% | 698 мс | 1633 мс |
| Qwen3-14B Q5_K_XL (thinking off) | 12 488 | 3.8 ГБ | **35.2** | 75.0% | 90.9% | 81.8% | **2027 мс** | 3363 мс |

**Плотная модель проигрывает по физике, а не по уму.** Qwen3-14B лучший по тулам (90.9%), но
35 tok/s против 105–120 у остальных: на каждый токен она читает все 9.8 ГБ весов, а MoE 30B-A3B —
только ~3B активных параметров. При 448 ГБ/с это ровно те цифры. Разреженная архитектура —
единственный способ получить низкую задержку в 16 ГБ.

**Урезание контекста не спасает:** 16k → 8k экономит 776 MiB (замер на Q3), потому что KV-кэш у
A3B мал. Выбор идёт по весам, не по контексту.

Влияние правки «машина в префиксе» (v1 → v2), точность вызова тулов:
gpt-oss 72.7 → 84.8 · Qwen Q3 72.7 → 78.8 · Qwen Q2 69.7 → 84.8 · Qwen3-14B 78.8 → 90.9.
Первая фраза при этом замедлилась у всех — агент перестал отделываться вопросом и пошёл в поиск.

### Что бенчмарк вскрыл в самом агенте

- **Запах топлива: провал у всех четырёх.** Промпт требует сначала сказать «прекрати ехать», но
  все модели начинают диагностировать. Это не различает кандидатов — это чинить промптом (день 6).
- Первый запрос к поиску стоит **~800 мс** — это разовая загрузка модели-эмбеддера. Прогревать
  при старте сервера, иначе первый вопрос водителя получит лишнюю секунду.
- Проверки на синонимы были слишком узкими: ответ «you should not drive the car» засчитывался как
  провал. Исправлено в `scenarios.yaml`.

### TTS

_(день 3, после LLM)_

### Сквозная задержка

_(день 6)_

## Материал для README

- Зачем такой стек: см. DECISIONS.md.
- Как замеряется задержка: client-side VAD (конец речи) → первый аудиосэмпл ответа в браузере; плюс серверная разбивка по этапам; benchmark-режим с фиксированными WAV → p50/p95.
- Реальный Torque Pro подключается без изменений кода: Torque → Settings → Data Logging & Upload → Webserver URL → `https://<backend>/torque`.
- Симулятор говорит с бэкендом по тому же HTTP-протоколу, что и телефон.
- Атрибуция: Stack Exchange контент — CC BY-SA, ссылка на каждый тред в ответах/карточках.
