# Mika.TTS.CosyVoice

Голосовой синтез Мики: TRT-LLM + TensorRT Flow + PyTorch HiFT, модель
Fun-CosyVoice3-0.5B. Форк https://github.com/Brakanier/FastCosyVoice (remote
`upstream`), наш `origin` — IceCookieAi/FastCosyVoice. Наши коммиты лежат
поверх upstream `95e51ab`. Каталог `cosyvoice/` — вендорный код CosyVoice:
не форматировать и не перестраивать попутно.

Общие правила Aiceon — в корневом `../AGENTS.md`: прочитай его до изменений,
он сам не подгружается при открытии этого репо отдельно.

Единственный клиент — Mika.TTS.Core (`app/synthesis/backends/cosy_voice.py`),
бэкенд ему назначает Hub (тумблер CosyVoice/OmniVoice, см. `Mika.TTS.Core/AGENTS.md`).

## Команды

Сервис живёт только в Docker, сборка и запуск ИЗ КОРНЯ Aiceon — код копируется
в образ (`COPY .`), правка кода требует пересборки:

```bash
docker compose build cosy
docker compose up -d cosy
docker compose up -d --force-recreate cosy   # после смены env в compose или новой сборки
```

Порт 8870. Playground и `/health` — `http://localhost:8870/`. Тестов в репо
нет; проверка правок — замер TTFA + детекторы качества ниже.

```bash
python scripts/tts_ttfa_bench.py   # из корня Aiceon; время до первого звука — главный критерий
```

## Конвейер и конфиг

Вход — `mika_server.py` (`POST /v1/tts/zero-shot`, raw PCM int16 LE mono
24 кГц без заголовка). Веса в `pretrained_models/` (junction на E:, движки TRT
собираются при первом старте). Боевая LLM — `llm.rl.pt`. Основные env (заданы
в корневом `docker-compose.yml`, сервис `cosy`): `COSYVOICE_MODEL_DIR`,
`COSYVOICE_LLM_MODEL_NAME`, `COSYVOICE_LOAD_TRT`, `COSYVOICE_LOAD_TRT_LLM`,
`COSYVOICE_TRT_LLM_DTYPE`, `COSYVOICE_TRT_LLM_KV_CACHE_TOKENS`,
`COSYVOICE_LAZY_SPEAKER_REGISTRATION`, `COSYVOICE_FREQ_PENALTY`,
`COSYVOICE_REPETITION_PENALTY`, `COSYVOICE_MIN_TOKENS_RATIO`,
`VOICE_REFERENCE_NAME`. Референсы — общее хранилище `Mika.TTS.Core/references`
(смонтировано read-only).

## История дефектов качества голоса

Владелец держит свои коммиты поверх форка из-за редких, но критичных
дефектов, которые трудно поймать и воспроизвести:

- **срыв тона** — женский референс, а голос звучит мужским/низким (mode-switch);
- **залипание в тишину** — модель генерирует тишину вместо речи;
- **обрезка конца** фразы на длинных текстах.

Лечились сэмплингом TRT-LLM: RAS v3 logits processor (`fastcosyvoice/ras_processor.py`),
`repetition_penalty=1.0`, `min_tokens_ratio=1.5`, `safe_pad_id=true`,
`frequency_penalty=0.0`, `T=1.0/top_p=0.8/top_k=25`, модель `llm.rl.pt`.
Разборы — `tmp/plans/` (`HANDOFF.md`, `mode-switch-*`, `cutoff-relax-*`),
детекторы — `tmp/research/` (`final_ab.py` — срыв тона по pyin, `cutoff_test.py`).
`tmp/` в `.gitignore` — файлы локальные, не в репо.

Известный открытый дефект (есть и до правок 03.10.2026): короткая фраза
«Слушай, погоди-ка.» на референсе `bright/2` залипает в тишину до 9–11 с
примерно в 4% синтезов (4/100 на базе). Текущие правки его не трогают.

## Откат оптимизаций 03.10.2026

Состояние ДО всех правок 03.10.2026 — коммит `9a1d1be` («uv из официального
образа вместо curl | sh»). Три правки TTFA идут отдельными коммитами поверх
него, откатываются по одному через `git revert`:

| коммит | что | связь с дефектами |
|---|---|---|
| `2f48e66` | первый чанк Flow 15 токенов вместо 25 | трогает только нарезку Flow; срыв тона 4/120 до и после |
| `d02689d` | Flow+HiFT на своём CUDA-потоке | звук побайтно тот же (seed, use_ras=false) — к дефектам отношения не имеет |
| `3d6bc20` | speech tokenizer на CPU, референсы регистрируются на старте (`COSYVOICE_LAZY_SPEAKER_REGISTRATION=false`) | меняет 1–2 токена референса из сотен в 7 из 19 референсов; эти токены идут в промпт LLM — первый подозреваемый, если вернётся срыв тона или тишина |

Порядок при жалобе на тишину или мужской тон: сначала `git revert 3d6bc20`,
пересобрать, проверить; не помогло — `git revert 2f48e66`; полный откат —
`git revert 3d6bc20 d02689d 2f48e66` (или временно `git checkout 9a1d1be`).
После отката: из корня Aiceon `docker compose build cosy && docker compose
up -d --force-recreate cosy`.

Корневой compose задаёт `COSYVOICE_LAZY_SPEAKER_REGISTRATION: "false"`; с
откатанным `3d6bc20` это работает (токенизатор снова на GPU, регистрация на
старте), менять не обязательно. Тишина на коротких фразах на `bright/2` была
и до правок — откат её не вылечит.
