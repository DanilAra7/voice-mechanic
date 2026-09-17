# Проект: голосовой агент «автомеханик»

## ТЗ (от лида)

- Голосовой агент на **открытой модели** (никаких LLM API).
- Запуск на сервере с ограничением GPU. Внутреннее ограничение проекта: **весь стек ≤ 16 ГБ VRAM**.
- **Минимизировать задержку.**
- Именно **агент с несколькими тулами**, не «таск-трекер».
- Только **английский** язык.
- Модель — **максимально умная**, что влезает (квантизация допустима).
- Сдаём: **сайт**, где пользователь говорит с агентом и **сам может замерить латентность**.
- Дедлайн: **2026-09-24** (неделя с 2026-09-17).

## Ограничения и вводные

- Сервера нет → Vast.ai почасово (рекомендация лида), карта ровно 16 ГБ (физически гарантирует лимит).
- Локальная машина: MacBook Air M4, 16 ГБ unified memory, macOS 26.6. Есть `uv`, `node 24`, `git`. Нет `brew`, `docker`, `ollama`.
- Реальная машина пользователя: **Audi A4** (есть Android + Torque Pro + OBD-адаптер), но в демо используем **только симулятор**.
- Лид пользователю лично незнаком → всё должно работать «из коробки» и быть понятно без объяснений.

## Архитектура (черновик)

```
Браузер (статический фронт: Cloudflare Pages / GitHub Pages)
  ├─ микрофон → WebSocket (аудио + события/метрики)
  ├─ панель разговора, карточки tool-calls
  ├─ Garage: выбор авто, инъекция неисправностей, графики датчиков
  └─ Latency: client-side VAD → первый звук ответа; разбивка по этапам; benchmark-режим (WAV-фразы), p50/p95; RTT; VRAM
        │ Cloudflare Tunnel (HTTPS, без домена)
        ▼
Vast.ai инстанс (16 ГБ GPU)
  Pipecat pipeline: Silero VAD → Smart-Turn → ASR (Parakeet, CPU) → LLM → TTS (GPU)
  LLM-сервер: vLLM или llama.cpp (решится бенчмарком)
  Tools API + симулятор Torque + локальный поисковый индекс
```

## Компоненты и кандидаты

| Компонент | Кандидаты | Статус |
|---|---|---|
| LLM | gpt-oss-20b (MXFP4) · Qwen3-30B-A3B-Instruct-2507 (~Q3 или Q4 + `--n-cpu-moe`) · Qwen3-14B AWQ | бенчмарк, день 3 |
| TTS (цель — качество ~ElevenLabs) | Chatterbox / Chatterbox Turbo · Kyutai TTS 1.6B · Orpheus 3B (квант.) · VibeVoice-Realtime-0.5B · Qwen3-TTS (?) | A/B + TTFA, день 3 |
| ASR | Parakeet TDT 0.6B через sherpa-onnx на CPU | план |
| VAD / turn | Silero VAD, Pipecat Smart-Turn | план |
| Оркестрация | Pipecat, транспорт WebSocket (Cloudflare Tunnel не пропускает UDP → не WebRTC) | план |
| Поиск | гибрид BM25 + эмбеддинги (CPU), фильтр по make/model/year | план |
| Локальная LLM для разработки | Qwen3-8B / 4B Q4 (MLX или llama.cpp) — gpt-oss-20b на 16 ГБ Mac слишком тесно | план |

## Тулы агента

| Тул | Источник данных |
|---|---|
| `vehicle_profile` (get/set) | состояние сессии: make, model, year, mileage |
| `get_live_sensors` / `get_sensor_trend` | симулятор в формате Torque web-upload → SQLite |
| `lookup_dtc` | локальная база OBD-II кодов (открытый датасет, проверить лицензию) |
| `search_howto` | спарсенный carcarekiosk.com |
| `search_known_issues` | спарсенный startmycar.com |
| `search_forum` | **mechanics.stackexchange.com** из дампа (21k Q&A-тредов) |

Во время медленных тулов агент говорит фразу-заглушку («Let me check…»). Системный промпт содержит safety-оговорку (тормоза, рулевое, запах топлива → не ехать, в сервис).

## Автомобили для MVP (5 шт.)

| id | Машина | Двигатель |
|---|---|---|
| `audi_a4_b8` | Audi A4 B8, 2009–2016 (машина пользователя) | 2.0 TFSI I4 turbo |
| `honda_accord_9` | Honda Accord 9th gen, 2013–2017 | 2.4 I4 |
| `ford_f150_13` | Ford F-150 13th gen, 2015–2020 | 2.7 EcoBoost V6 |
| `honda_civic_10` | Honda Civic 10th gen, 2016–2021 | 2.0 I4 |
| `toyota_corolla_11` | Toyota Corolla 11th gen, 2014–2019 | 1.8 I4 |

Выбраны по объёму данных (см. DECISIONS.md, NOTES.md → «Источники данных»).
