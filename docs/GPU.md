# Runbook: аренда GPU и бенчмарки (день 3)

Цель дня: выбрать связку **LLM + TTS**, которая влезает в **16 ГБ VRAM** и даёт минимальную задержку, и подтвердить выбор цифрами.

## 1. Аренда инстанса на Vast.ai

- Фильтр: **GPU RAM = 16 GB** (RTX 4060 Ti 16GB / RTX 4080 / RTX 5070 Ti / A4000), ≥ 8 vCPU, ≥ 32 ГБ RAM, диск ≥ 60 ГБ, `Direct` соединение (для SSH и портов).
- Образ: `pytorch/pytorch` или образ с CUDA 12.x; ставим всё сами через `uv`.
- Карта ровно на 16 ГБ — это и есть доказательство лимита: больше физически не влезет.
- **Сразу после старта** запускаем автоостановку, чтобы не платить за забытый инстанс:
  ```bash
  CONTAINER_ID=$CONTAINER_ID VAST_API_KEY=... nohup python scripts/idle_shutdown.py --idle-minutes 20 &
  ```
- В конце сессии: `vastai stop instance $CONTAINER_ID` (или кнопка Stop) и запись в `docs/PROGRESS.md` → раздел Vast.ai.

## 2. Доставка кода и данных

Код — через git (репозиторий на GitHub) или `rsync -avz --exclude data/raw ./ root@<host>:/workspace/mechanic/`.

Индекс и обработанные данные (`data/processed`, `data/index`) не в git, поэтому либо копируем `rsync`-ом (~200 МБ), либо пересобираем на инстансе:
```bash
uv run python -m mechanic.data.stackexchange && uv run python -m mechanic.knowledge.search build
```

## 3. Кандидаты LLM

Замеряем на одинаковых 32 сценариях (`uv run python -m mechanic.evals.run`).

| # | Модель | Движок | Команда (черновик, уточнить по факту) |
|---|---|---|---|
| 1 | gpt-oss-20b (MXFP4) | vLLM | `vllm serve openai/gpt-oss-20b --port 8001 --gpu-memory-utilization 0.75 --max-model-len 16384` |
| 2 | Qwen3-30B-A3B-Instruct-2507, ~Q3 | llama.cpp | `llama-server -hf <repo>:UD-Q3_K_XL --port 8001 -c 16384 -ngl 99 --jinja` |
| 3 | Qwen3-30B-A3B-Instruct-2507, Q4 + эксперты на CPU | llama.cpp | то же + `--n-cpu-moe <N>` |
| 4 | Qwen3-14B AWQ | vLLM | `vllm serve <repo> --quantization awq --port 8001 --gpu-memory-utilization 0.6` |

Для каждого фиксируем: **VRAM (пик)**, **TTFT**, **токенов/с**, **tool accuracy**, **content accuracy**, **p50/p95 первой фразы** — таблица идёт в `docs/NOTES.md` → «Бенчмарки» и потом в README.

Важно: `--gpu-memory-utilization` подбирается так, чтобы TTS и (если понадобится) ASR поместились рядом.

## 4. Кандидаты TTS

Один и тот же набор из 10 реплик механика, замеряем:
- **TTFA** — время до первого аудиочанка (главное для задержки);
- **RTF** — во сколько раз быстрее реального времени;
- **VRAM**;
- качество — слепое сравнение на слух (файлы кладём в `data/cache/tts_samples/`).

Кандидаты: Chatterbox / Chatterbox Turbo, Kyutai TTS 1.6B, Orpheus 3B (квантованная), VibeVoice-Realtime-0.5B, Qwen3-TTS. Перед запуском проверить, что вышло нового.

## 5. Критерий выбора

Берём самую умную LLM, у которой:
- весь стек (LLM + TTS + буферы) ≤ 16 ГБ,
- p50 первой фразы ≤ ~500 мс на тексте (без учёта ASR и TTS),
- tool accuracy ≥ 90% на сценариях.

Если ни одна не проходит — снижаем контекст до 8k и повторяем; следующий шаг — Qwen3-14B AWQ как запасной.
