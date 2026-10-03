"""FastAPI server для Mika.TTS.Core ↔ FastCosyVoice3 integration.

Контракт (зеркало `Mika.TTS.Core/app/services/tts/cosy_voice_engine.py`):

  POST /v1/tts/zero-shot   JSON {text, reference_id="pool/N", streaming,
                                 temperature, top_p, top_k}
                           → stream raw PCM int16 LE mono @ 24 kHz, без RIFF.
  GET  /health             → {status, sample_rate, trt_llm_loaded, ...}
  GET  /v1/references      → список доступных pool/N для отладки.

Reference resolution (зеркало ReferenceManager в TTS.Core):
  references/{pool}/{N}/{voice_name}.{mp3|wav|flac|...}  ← аудио
  references/{pool}/{N}/{voice_name}.lab                  ← транскрипция (UTF-8)
  voice_name = env VOICE_REFERENCE_NAME (дефолт "mita")
  prompt_text = "You are a helpful assistant.<|endofprompt|>{transcription}"
  (CosyVoice3-специфичный формат, см. run_fast.py:load_prompt_text)

Зачем нужно:
  - run_fast.py — CLI, не HTTP. Апстримный runtime/python/fastapi/server.py
    привязан к старому CosyVoice/CosyVoice2 классу, а не FastCosyVoice3 с
    TRT-LLM + параллельным пайплайном.
  - Поэтому пишем свой минимальный сервер, который использует
    `FastCosyVoice3.inference_zero_shot_stream()` напрямую.
  - Speaker embeddings кэшируются через `add_zero_shot_spk()` — на втором
    запросе с тем же reference_id пропускаем re-encoding промпт-аудио.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Apстрим FastCosyVoice ожидает Matcha-TTS в sys.path (см. run_fast.py:24).
# Пустой каталог уже создан в Dockerfile; matcha-tts пакет идёт из PyPI.
sys.path.append("third_party/Matcha-TTS")

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from fastcosyvoice import FastCosyVoice3


# ----------------------------------------------------------------------------
# Конфиг через env (всё дефолтное — то же что в run_fast.py).
# ----------------------------------------------------------------------------

MODEL_DIR = os.environ.get(
    "COSYVOICE_MODEL_DIR",
    "pretrained_models/Fun-CosyVoice3-0.5B",
)
REFERENCES_DIR = Path(
    os.environ.get(
        "COSYVOICE_REFERENCES_DIR",
        "/workspace/references",
    )
).resolve()
VOICE_REFERENCE_NAME = os.environ.get("VOICE_REFERENCE_NAME", "mita")

# Default: llm.rl.pt (RL fine-tuned). Verified 2026-05-23 to produce significantly
# fewer "speaker mode-switch" defects on long pause-heavy Russian texts vs llm.pt
# (T2 esli-trigger 13%→3%, prod text 3-7%→0%). Voice quality unchanged. llm.pt
# stays as fallback — pass COSYVOICE_LLM_MODEL_NAME=llm.pt to revert.
LLM_MODEL_NAME = os.environ.get("COSYVOICE_LLM_MODEL_NAME", "llm.rl.pt")
USE_TRT_FLOW = os.environ.get("COSYVOICE_LOAD_TRT", "true").lower() == "true"
USE_TRT_LLM = os.environ.get("COSYVOICE_LOAD_TRT_LLM", "true").lower() == "true"
TRT_LLM_DTYPE = os.environ.get("COSYVOICE_TRT_LLM_DTYPE", "bfloat16")
TRT_LLM_KV_CACHE_TOKENS = int(
    os.environ.get("COSYVOICE_TRT_LLM_KV_CACHE_TOKENS", "3072")
)
INSTRUCTION = os.environ.get(
    "COSYVOICE_INSTRUCTION",
    "You are a helpful assistant. Young girl, female voice",
)
# VRAM optimization flags
EMPTY_CACHE_BETWEEN_REQUESTS = (
    os.environ.get("COSYVOICE_EMPTY_CACHE_BETWEEN_REQUESTS", "true").lower() == "true"
)
LAZY_SPEAKER_REGISTRATION = (
    os.environ.get("COSYVOICE_LAZY_SPEAKER_REGISTRATION", "false").lower() == "true"
)

# Silence-hallucination fix defaults (Layer 1, root-fix via TRT-LLM sampling).
# Per-request override via TTSRequest fields below.
# See tmp/plans/cosy-silence-fix.md.
FREQ_PENALTY_DEFAULT = float(os.environ.get("COSYVOICE_FREQ_PENALTY", "0.3"))
# NO_REPEAT_NGRAM was set to 2 originally to prevent silent-token loops in the
# silence-hallucination fix. Investigation 2026-05-22 showed it CAUSES
# speaker mode-switch defects on long pause-heavy texts (55% defect rate on
# "Если"-style text vs 20% without). Silence regression test (75 generations
# on anxious/2) confirmed silence stays fixed at 0% with nrg=0 — other params
# (freq_pen, safe_pad, min_tokens_ratio) carry the silence fix. Default
# changed 2 → 0.
NO_REPEAT_NGRAM_DEFAULT = int(os.environ.get("COSYVOICE_NO_REPEAT_NGRAM", "0"))
SAFE_PAD_ID_DEFAULT = os.environ.get("COSYVOICE_SAFE_PAD_ID", "true").lower() == "true"
MIN_TOKENS_RATIO_DEFAULT = float(os.environ.get("COSYVOICE_MIN_TOKENS_RATIO", "1.5"))
# repetition_penalty: was hardcoded 1.1 in cosyvoice.py. Accumulates over FULL
# context. On long texts (250+ chars) it suppresses common speech tokens, making
# EOS relatively attractive → premature cutoff of last words. Switched default
# to 1.0 (off) on 2026-05-29. RAS v3 handles repetition защиту через sliding
# window — глобальный rep_penalty избыточен и контрпродуктивен для TTS длинных
# фраз. Per-request override via TTSRequest.
REPETITION_PENALTY_DEFAULT = float(os.environ.get("COSYVOICE_REPETITION_PENALTY", "1.0"))
# Disable Cosy's internal sentence splitter. Default TRUE because TTS.Core's
# RealtimeTTS already pre-splits the stream into sentence chunks; running
# Cosy's split_paragraph on top is redundant (30-50ms × N extra inferences)
# and introduces prosody jumps at boundaries. Set false to restore upstream.
DISABLE_TEXT_SPLIT_DEFAULT = (
    os.environ.get("COSYVOICE_DISABLE_TEXT_SPLIT", "true").lower() == "true"
)
# RAS (Repetition-Aware Sampling) custom logits processor — v3 fix verified
# 2026-05-23: eliminated ALL mode-switch defects in 90-generation A/B test
# (0/30 on 3 hard texts vs 3/90 without). Default ON. Adds ~50-75 logit-processor
# calls per generation (Python overhead ~1-3ms each, mostly hidden by GPU work).
USE_RAS_DEFAULT = os.environ.get("COSYVOICE_USE_RAS", "true").lower() == "true"

# Расширения аудио в порядке приоритета — совпадает с ReferenceManager
# из Mika.TTS.Core, чтобы файлы в общей папке references/ резолвились
# идентично.
AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".aiff")
TEXT_SUFFIX = ".lab"

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8870"))

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mika_server")


# ----------------------------------------------------------------------------
# Глобальное состояние модели (load once, share across requests).
# ----------------------------------------------------------------------------


class ServerState:
    cosyvoice: Optional[FastCosyVoice3] = None
    registered_refs: set[str] = set()
    sample_rate: int = 24000
    trt_llm_loaded: bool = False


state = ServerState()


def _resolve_reference(reference_id: str) -> tuple[Path, Path, str]:
    """``"pool/N"`` → (audio_path, lab_path, spk_id). Кидает HTTPException."""
    if "/" not in reference_id:
        raise HTTPException(400, f"reference_id must be 'pool/N', got {reference_id!r}")
    pool, version = reference_id.split("/", 1)
    pool = pool.strip()
    version = version.strip()
    if not pool or not version:
        raise HTTPException(
            400, f"reference_id has empty pool/version: {reference_id!r}"
        )

    version_dir = REFERENCES_DIR / pool / version
    if not version_dir.is_dir():
        raise HTTPException(404, f"reference dir not found: {version_dir}")

    audio_path: Optional[Path] = None
    for ext in AUDIO_EXTENSIONS:
        candidate = version_dir / f"{VOICE_REFERENCE_NAME}{ext}"
        if candidate.exists():
            audio_path = candidate
            break
    if audio_path is None:
        raise HTTPException(
            422,
            f"audio not found: {version_dir}/{VOICE_REFERENCE_NAME}.<{','.join(AUDIO_EXTENSIONS)}>",
        )

    lab_path = version_dir / f"{VOICE_REFERENCE_NAME}{TEXT_SUFFIX}"
    if not lab_path.exists():
        raise HTTPException(422, f"transcription not found: {lab_path}")

    spk_id = f"{pool}__{version}"  # внутренний ID для add_zero_shot_spk кэша
    return audio_path, lab_path, spk_id


def _build_prompt_text(lab_path: Path) -> str:
    transcription = lab_path.read_text(encoding="utf-8").strip()
    if not transcription:
        raise HTTPException(422, f"empty transcription: {lab_path}")
    return f"{INSTRUCTION}<|endofprompt|>{transcription}"


def _ensure_speaker_registered(reference_id: str) -> tuple[Path, str, str]:
    """Регистрирует speaker embedding для reference_id если ещё не было.

    Возвращает (audio_path, prompt_text, spk_id) для последующего inference.
    """
    audio_path, lab_path, spk_id = _resolve_reference(reference_id)
    prompt_text = _build_prompt_text(lab_path)
    if spk_id not in state.registered_refs:
        t0 = time.perf_counter()
        state.cosyvoice.add_zero_shot_spk(prompt_text, str(audio_path), spk_id)
        state.registered_refs.add(spk_id)
        logger.info(
            "registered speaker %s (audio=%s, %.0fms)",
            spk_id,
            audio_path.name,
            (time.perf_counter() - t0) * 1000,
        )
    return audio_path, prompt_text, spk_id


# ----------------------------------------------------------------------------
# Lifespan: грузим модель один раз на старте.
# ----------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Loading FastCosyVoice3 from %s (TRT-LLM=%s, TRT-Flow=%s, dtype=%s)",
        MODEL_DIR,
        USE_TRT_LLM,
        USE_TRT_FLOW,
        TRT_LLM_DTYPE,
    )
    t0 = time.perf_counter()
    state.cosyvoice = FastCosyVoice3(
        model_dir=MODEL_DIR,
        fp16=True,
        load_trt=USE_TRT_FLOW,
        load_trt_llm=USE_TRT_LLM,
        trt_llm_dtype=TRT_LLM_DTYPE,
        trt_llm_kv_cache_tokens=TRT_LLM_KV_CACHE_TOKENS,
        llm_model_name=LLM_MODEL_NAME,
    )
    state.sample_rate = state.cosyvoice.sample_rate
    state.trt_llm_loaded = bool(getattr(state.cosyvoice, "trt_llm_loaded", False))
    load_secs = time.perf_counter() - t0
    logger.info(
        "FastCosyVoice3 ready in %.1fs | sample_rate=%d, trt_llm_loaded=%s",
        load_secs,
        state.sample_rate,
        state.trt_llm_loaded,
    )

    # Register speakers согласно LAZY_SPEAKER_REGISTRATION flag.
    # Lazy=true: только проверяем что dir существует и пишем сколько pool/N
    # доступно — конкретные spk_id регистрируются на первом запросе с этим
    # reference_id. Экономит VRAM ~50-100 MB и время старта (модель грузится
    # за 4-5с вместо 20с при ~100 prelocaded speakers).
    if not REFERENCES_DIR.is_dir():
        logger.warning(
            "references dir not found: %s (server will reject all requests)",
            REFERENCES_DIR,
        )
    elif LAZY_SPEAKER_REGISTRATION:
        available = sum(
            1
            for pool_dir in REFERENCES_DIR.iterdir()
            if pool_dir.is_dir()
            for version_dir in pool_dir.iterdir()
            if version_dir.is_dir()
        )
        logger.info(
            "Lazy speaker registration enabled: %d pool/N entries available in %s",
            available,
            REFERENCES_DIR,
        )
    else:
        pre_registered = 0
        for pool_dir in sorted(REFERENCES_DIR.iterdir()):
            if not pool_dir.is_dir():
                continue
            for version_dir in sorted(pool_dir.iterdir()):
                if not version_dir.is_dir():
                    continue
                reference_id = f"{pool_dir.name}/{version_dir.name}"
                try:
                    _ensure_speaker_registered(reference_id)
                    pre_registered += 1
                except HTTPException as exc:
                    logger.warning("skip %s: %s", reference_id, exc.detail)
        logger.info(
            "Pre-registered %d speakers from %s",
            pre_registered,
            REFERENCES_DIR,
        )

    # Warmup — короткий прогон чтобы CUDA kernel'ы засинхронились.
    # Используем первый pool/N из refs/. В lazy-режиме это автоматически
    # регистрирует один speaker (а не все 100).
    if state.cosyvoice and REFERENCES_DIR.is_dir():
        first_ref: Optional[str] = None
        for pool_dir in sorted(REFERENCES_DIR.iterdir()):
            if not pool_dir.is_dir():
                continue
            for version_dir in sorted(pool_dir.iterdir()):
                if version_dir.is_dir():
                    first_ref = f"{pool_dir.name}/{version_dir.name}"
                    break
            if first_ref:
                break
        if first_ref:
            try:
                t0 = time.perf_counter()
                audio_path, prompt_text, spk_id = _ensure_speaker_registered(first_ref)
                for _ in state.cosyvoice.inference_zero_shot_stream(
                    tts_text="Прогрев.",
                    prompt_text=prompt_text,
                    prompt_wav=str(audio_path),
                    zero_shot_spk_id=spk_id,
                ):
                    pass
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    # Освобождаем PyTorch caching pool после warmup — это
                    # baseline для дальнейшего наблюдения роста VRAM.
                    if EMPTY_CACHE_BETWEEN_REQUESTS:
                        torch.cuda.empty_cache()
                logger.info(
                    "Warmup on %s done in %.2fs", first_ref, time.perf_counter() - t0
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Warmup failed: %s", exc)

    yield

    logger.info("Shutting down")


app = FastAPI(title="Mika FastCosyVoice3 server", lifespan=lifespan)


# ----------------------------------------------------------------------------
# Endpoints.
# ----------------------------------------------------------------------------


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    reference_id: str = Field(..., description='"pool/N", e.g. "neutral/1"')
    streaming: bool = True
    # 2026-05-24: подняли T=0.7→1.0, top_p=0.85→0.8, top_k=20→25 чтобы matched
    # PyTorch upstream baseline (cosy-orig). Detector-based A/B 160 syntheses
    # показал defect rate 50%→10% на проблемных текстах. См.
    # tmp/plans/mode-switch-rootcause-2026-05-24.md.
    temperature: float = 1.0
    top_p: float = 0.8
    top_k: int = 25
    # Silence-hallucination fix overrides (None → server-side env defaults).
    # See tmp/plans/cosy-silence-fix.md. For A/B baseline, set all four to
    # legacy values: frequency_penalty=0.0, no_repeat_ngram_size=0,
    # safe_pad_id=False, min_tokens_ratio=2.0.
    frequency_penalty: float | None = Field(
        None,
        description="Accumulating penalty on repeated tokens (RAS-like). "
        "0.0 disables; 0.3 server default.",
    )
    no_repeat_ngram_size: int | None = Field(
        None,
        description="Blocks n-gram repetition in TRT-LLM. 0 disables; "
        "2 server default.",
    )
    safe_pad_id: bool | None = Field(
        None,
        description="True → pad_id=sos_token_id (fix); False → pad_id=eos1 "
        "(legacy, TRT-LLM issue #280). Default True.",
    )
    min_tokens_ratio: float | None = Field(
        None,
        description="Multiplier for text_tokens → min_tokens. 2.0 legacy, "
        "1.0 default (Russian-friendly).",
    )
    disable_text_split: bool | None = Field(
        None,
        description="True = skip Cosy internal sentence-split (caller pre-"
        "splits, e.g. RealtimeTTS). Default true. False restores "
        "upstream split_paragraph behavior.",
    )
    random_seed: int | None = Field(
        None,
        description="Pin TRT-LLM random_seed for deterministic generation. "
        "For investigation of voice-mode-switch defect — same "
        "seed must produce byte-identical output. None = "
        "TRT-LLM internal randomness.",
    )
    bad_speech_ids: list[int] | None = Field(
        None,
        description="Speech token IDs (0..6560) to block via TRT-LLM "
        "bad_words_list. Used to prevent voice mode-switch "
        "by blocking suspect low-pitch trigger tokens.",
    )
    repetition_penalty: float | None = Field(
        None,
        description="TRT-LLM repetition_penalty (default 1.1). Acts on "
        "FULL token history. Higher = harder to repeat any "
        "previously-used token. Testing 1.3-1.5 to see if it "
        "breaks low-pitch attractor lock-in.",
    )
    use_ras: bool | None = Field(
        None,
        description="Enable Repetition-Aware Sampling logits processor "
        "(VALL-E 2 RAS port). Custom-implemented for TRT-LLM "
        "since native path lacks it. Eliminates speaker "
        "mode-switch defects in pause-heavy texts. Default "
        "TRUE (server env COSYVOICE_USE_RAS). Set false to "
        "disable per-request.",
    )
    instruction: str | None = Field(
        None,
        description="Override system instruction prepended to prompt (before "
        "<|endofprompt|>{transcription}). When set, bypasses "
        "speaker-embedding cache (re-encodes reference each call, "
        "+~30-50ms). Empty string = no instruction text. None = "
        "use server-default COSYVOICE_INSTRUCTION env (cached path). "
        "Use for playground experiments — TTS.Core does NOT need "
        "to send this; default behavior is unchanged.",
    )


PLAYGROUND_HTML = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>FastCosyVoice3 playground</title>
<style>
  :root { color-scheme: dark; }
  body {
    font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
    max-width: 760px; margin: 24px auto; padding: 0 16px;
    background: #0f1115; color: #e6e6e6;
  }
  h1 { font-size: 18px; margin: 0 0 4px; }
  .sub { color: #888; font-size: 13px; margin-bottom: 18px; }
  label { display: block; margin: 10px 0 4px; font-size: 13px; color: #bbb; }
  textarea, select, input[type=number] {
    width: 100%; box-sizing: border-box; padding: 8px 10px; border-radius: 6px;
    background: #1a1d24; color: #e6e6e6; border: 1px solid #2c313b;
    font-family: inherit; font-size: 14px;
  }
  textarea { min-height: 80px; resize: vertical; }
  .row { display: grid; grid-template-columns: 2fr 1fr 1fr 1fr; gap: 10px; align-items: end; }
  .row3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; align-items: end; }
  .section-hdr { color: #555; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; margin: 16px 0 4px; }
  button {
    margin-top: 14px; padding: 9px 18px; border: 0; border-radius: 6px;
    background: #4f8cff; color: white; font-weight: 600; cursor: pointer;
  }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  audio { width: 100%; margin-top: 12px; }
  .metrics {
    margin-top: 12px; padding: 10px 12px; background: #1a1d24;
    border-radius: 6px; font-family: ui-monospace, monospace;
    font-size: 13px; color: #aaa; white-space: pre-wrap;
  }
  .err { color: #ff7a7a; }
  .ok  { color: #7aff9d; }
</style>
</head>
<body>
<h1>FastCosyVoice3 — playground</h1>
<div class="sub">model: <span id="model">…</span> · sr: <span id="sr">24000</span> Hz · TRT-LLM: <span id="trt">…</span></div>

<label>Текст</label>
<textarea id="text">Привет, это полноценный сервер на пятитысячной картe.</textarea>

<label>
  Instruction
  <span style="color:#888; font-weight:normal;">— стиль/эмоция для модели. Сервер сам обернёт в правильный формат
  <code style="background:#222;padding:1px 4px;border-radius:3px;">"You are a helpful assistant. &lt;твоя инструкция&gt; &lt;|endofprompt|&gt;"</code>.
  EN-инструкция на русском тексте может дать акцент — пробуй ту же фразу на русском.
  Эффект слабый из-за сильного reference. Пусто = быстрый кэш-путь; любой текст = bypass кэша (+30-50ms). TTS.Core эту фичу не использует.</span>
</label>
<textarea id="instruction" rows="2" placeholder="(empty → use server default 'You are a helpful assistant.')"></textarea>

<div class="row">
  <div>
    <label>reference_id</label>
    <select id="refid"></select>
  </div>
  <div>
    <label>temperature</label>
    <input id="temp" type="number" step="0.05" min="0.1" max="1.5" value="0.7">
  </div>
  <div>
    <label>top_p</label>
    <input id="top_p" type="number" step="0.01" min="0.1" max="1.0" value="0.85">
  </div>
  <div>
    <label>top_k</label>
    <input id="top_k" type="number" step="1" min="1" max="200" value="20">
  </div>
</div>

<div class="section-hdr">Sampling overrides — пусто = серверный default</div>
<div class="row3">
  <div>
    <label>frequency_penalty <span id="def_fp_lbl" style="color:#555"></span></label>
    <input id="freq_penalty" type="number" step="0.05" min="0.0" max="2.0" placeholder="(server default)">
  </div>
  <div>
    <label>no_repeat_ngram_size <span id="def_nrg_lbl" style="color:#555"></span></label>
    <input id="no_repeat_ngram" type="number" step="1" min="0" max="10" placeholder="(server default)">
  </div>
  <div>
    <label>min_tokens_ratio <span id="def_mtr_lbl" style="color:#555"></span></label>
    <input id="min_tokens_ratio" type="number" step="0.1" min="0.1" max="3.0" placeholder="(server default)">
  </div>
</div>

<button id="go">Synthesize</button>

<audio id="audio" controls></audio>
<div class="metrics" id="metrics">—</div>

<script>
const SR = 24000;

function pcmToWav(pcm, sampleRate) {
  // Раз PCM s16le mono — оборачиваем в WAV: 44-байтный RIFF + data.
  const buf = new ArrayBuffer(44 + pcm.byteLength);
  const view = new DataView(buf);
  const wr = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  wr(0, 'RIFF');
  view.setUint32(4, 36 + pcm.byteLength, true);
  wr(8, 'WAVE'); wr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);        // PCM
  view.setUint16(22, 1, true);        // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true);        // block align
  view.setUint16(34, 16, true);       // bits per sample
  wr(36, 'data');
  view.setUint32(40, pcm.byteLength, true);
  new Uint8Array(buf, 44).set(new Uint8Array(pcm));
  return new Blob([buf], { type: 'audio/wav' });
}

async function loadHealth() {
  const h = await (await fetch('/health')).json();
  document.getElementById('model').textContent = h.model_dir || '?';
  document.getElementById('sr').textContent = h.sample_rate || SR;
  document.getElementById('trt').textContent = h.trt_llm_loaded ? 'ON' : 'off';
  if (h.instruction_default) {
    document.getElementById('instruction').placeholder =
      '(empty → server default: "' + h.instruction_default + '")';
  }
  if (h.freq_penalty_default !== undefined)
    document.getElementById('def_fp_lbl').textContent = '(def: ' + h.freq_penalty_default + ')';
  if (h.no_repeat_ngram_default !== undefined)
    document.getElementById('def_nrg_lbl').textContent = '(def: ' + h.no_repeat_ngram_default + ')';
  if (h.min_tokens_ratio_default !== undefined)
    document.getElementById('def_mtr_lbl').textContent = '(def: ' + h.min_tokens_ratio_default + ')';
}
async function loadRefs() {
  const r = await (await fetch('/v1/references')).json();
  const sel = document.getElementById('refid');
  sel.innerHTML = '';
  for (const ref of (r.references || [])) {
    if (!ref.audio || !ref.lab) continue;
    const opt = document.createElement('option');
    opt.value = ref.reference_id;
    opt.textContent = ref.reference_id + (ref.registered ? '' : '  (lazy)');
    if (ref.reference_id === 'neutral/1') opt.selected = true;
    sel.appendChild(opt);
  }
}
loadHealth().catch(e => document.getElementById('metrics').textContent = 'health failed: ' + e);
loadRefs().catch(e => document.getElementById('metrics').textContent = 'refs failed: ' + e);

const audio = document.getElementById('audio');
const metrics = document.getElementById('metrics');
const goBtn = document.getElementById('go');

goBtn.onclick = async () => {
  const body = {
    text: document.getElementById('text').value,
    reference_id: document.getElementById('refid').value,
    temperature: parseFloat(document.getElementById('temp').value),
    top_p: parseFloat(document.getElementById('top_p').value),
    top_k: parseInt(document.getElementById('top_k').value, 10),
  };
  const instrEl = document.getElementById('instruction');
  if (instrEl && instrEl.value.length > 0) body.instruction = instrEl.value;
  // Sampling overrides — include only when explicitly set (empty = server default)
  const fpEl = document.getElementById('freq_penalty');
  if (fpEl && fpEl.value !== '') body.frequency_penalty = parseFloat(fpEl.value);
  const nrgEl = document.getElementById('no_repeat_ngram');
  if (nrgEl && nrgEl.value !== '') body.no_repeat_ngram_size = parseInt(nrgEl.value, 10);
  const mtrEl = document.getElementById('min_tokens_ratio');
  if (mtrEl && mtrEl.value !== '') body.min_tokens_ratio = parseFloat(mtrEl.value);
  if (!body.text.trim() || !body.reference_id) return;
  goBtn.disabled = true;
  metrics.textContent = 'streaming…';
  metrics.className = 'metrics';

  const t0 = performance.now();
  let ttfb = null;
  try {
    const resp = await fetch('/v1/tts/zero-shot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const detail = await resp.text();
      throw new Error(resp.status + ' ' + detail.slice(0, 200));
    }
    const reader = resp.body.getReader();
    const chunks = [];
    let total = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (ttfb === null) ttfb = performance.now() - t0;
      chunks.push(value);
      total += value.byteLength;
    }
    const elapsed = performance.now() - t0;
    const pcm = new Uint8Array(total);
    let off = 0;
    for (const c of chunks) { pcm.set(c, off); off += c.byteLength; }
    const wav = pcmToWav(pcm.buffer, SR);
    audio.src = URL.createObjectURL(wav);
    audio.play();

    const dur = total / 2 / SR;
    const rtf = elapsed / 1000 / dur;
    const fpUsed  = body.frequency_penalty   !== undefined ? body.frequency_penalty   : '(srv)';
    const nrgUsed = body.no_repeat_ngram_size !== undefined ? body.no_repeat_ngram_size : '(srv)';
    const mtrUsed = body.min_tokens_ratio     !== undefined ? body.min_tokens_ratio     : '(srv)';
    metrics.className = 'metrics ok';
    metrics.textContent =
      `TTFB:     ${ttfb.toFixed(0)} ms\n` +
      `Total:    ${elapsed.toFixed(0)} ms\n` +
      `Audio:    ${dur.toFixed(2)} s  (${(total/1024).toFixed(0)} KiB PCM)\n` +
      `RTF:      ${rtf.toFixed(3)}  (×${(1/rtf).toFixed(1)} faster than realtime)\n` +
      `Params:   T=${body.temperature} P=${body.top_p} K=${body.top_k}  fp=${fpUsed}  nrg=${nrgUsed}  mtr=${mtrUsed}`;
  } catch (e) {
    metrics.className = 'metrics err';
    metrics.textContent = 'ERROR: ' + e.message;
  } finally {
    goBtn.disabled = false;
  }
};
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def playground() -> HTMLResponse:
    return HTMLResponse(PLAYGROUND_HTML)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok" if state.cosyvoice is not None else "loading",
        "sample_rate": state.sample_rate,
        "trt_llm_loaded": state.trt_llm_loaded,
        "registered_refs": sorted(state.registered_refs),
        "references_dir": str(REFERENCES_DIR),
        "voice_name": VOICE_REFERENCE_NAME,
        "model_dir": MODEL_DIR,
        # Defaults exposed so playground can show them in placeholders.
        "instruction_default": INSTRUCTION,
        "freq_penalty_default": FREQ_PENALTY_DEFAULT,
        "no_repeat_ngram_default": NO_REPEAT_NGRAM_DEFAULT,
        "safe_pad_id_default": SAFE_PAD_ID_DEFAULT,
        "min_tokens_ratio_default": MIN_TOKENS_RATIO_DEFAULT,
        "disable_text_split_default": DISABLE_TEXT_SPLIT_DEFAULT,
        "repetition_penalty_default": REPETITION_PENALTY_DEFAULT,
    }


@app.get("/v1/debug/memory")
async def debug_memory() -> dict:
    """Текущая GPU-память: PyTorch alloc/reserved + nvidia-smi total.

    Полезно мониторить:
      - allocated растёт но reserved стабильный → tight working set
      - reserved >> allocated → PyTorch держит большой кэш-пул
                                (помогает COSYVOICE_EMPTY_CACHE_BETWEEN_REQUESTS=true)
      - nvidia-smi >> reserved → не наш процесс жрёт, или TRT-LLM
                                  свой allocator вне torch
    """
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    return {
        "cuda_available": True,
        "device": torch.cuda.get_device_name(0),
        "torch_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
        "torch_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
        "torch_max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "torch_max_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        "registered_speakers": len(state.registered_refs),
        "config": {
            "kv_cache_tokens": TRT_LLM_KV_CACHE_TOKENS,
            "empty_cache_between_requests": EMPTY_CACHE_BETWEEN_REQUESTS,
            "lazy_speaker_registration": LAZY_SPEAKER_REGISTRATION,
            "pytorch_cuda_alloc_conf": os.environ.get(
                "PYTORCH_CUDA_ALLOC_CONF", "(default)"
            ),
        },
    }


@app.post("/v1/debug/empty_cache")
async def debug_empty_cache() -> dict:
    """Ручной вызов torch.cuda.empty_cache() — для замеров."""
    if not torch.cuda.is_available():
        return {"ok": False, "reason": "no cuda"}
    before_alloc = torch.cuda.memory_allocated()
    before_reserv = torch.cuda.memory_reserved()
    torch.cuda.empty_cache()
    after_alloc = torch.cuda.memory_allocated()
    after_reserv = torch.cuda.memory_reserved()
    return {
        "ok": True,
        "freed_reserved_gb": round((before_reserv - after_reserv) / 1024**3, 3),
        "allocated_gb_before": round(before_alloc / 1024**3, 3),
        "allocated_gb_after": round(after_alloc / 1024**3, 3),
        "reserved_gb_before": round(before_reserv / 1024**3, 3),
        "reserved_gb_after": round(after_reserv / 1024**3, 3),
    }


@app.get("/v1/references")
async def list_references() -> dict:
    """Список доступных pool/N — для отладки и проверки маунтов."""
    if not REFERENCES_DIR.is_dir():
        return {"references": [], "dir": str(REFERENCES_DIR), "exists": False}
    result = []
    for pool_dir in sorted(REFERENCES_DIR.iterdir()):
        if not pool_dir.is_dir():
            continue
        for version_dir in sorted(pool_dir.iterdir()):
            if not version_dir.is_dir():
                continue
            ref_id = f"{pool_dir.name}/{version_dir.name}"
            audio = None
            for ext in AUDIO_EXTENSIONS:
                cand = version_dir / f"{VOICE_REFERENCE_NAME}{ext}"
                if cand.exists():
                    audio = cand.name
                    break
            lab = version_dir / f"{VOICE_REFERENCE_NAME}{TEXT_SUFFIX}"
            result.append(
                {
                    "reference_id": ref_id,
                    "audio": audio,
                    "lab": lab.name if lab.exists() else None,
                    "registered": f"{pool_dir.name}__{version_dir.name}"
                    in state.registered_refs,
                }
            )
    return {"references": result, "dir": str(REFERENCES_DIR), "exists": True}


def _stream_pcm(
    text: str,
    prompt_text: str,
    audio_path: Path,
    spk_id: str,
    temperature: float,
    top_p: float,
    top_k: int,
    frequency_penalty: float | None,
    no_repeat_ngram_size: int | None,
    safe_pad_id: bool | None,
    min_tokens_ratio: float | None,
    disable_text_split: bool | None,
    random_seed: int | None,
    bad_speech_ids: list[int] | None,
    repetition_penalty: float | None,
    use_ras: bool,
):
    """Generator: yields raw PCM int16 LE mono @ 24 kHz, без RIFF.

    Запускается в worker-потоке через StreamingResponse — FastCosyVoice3
    generator блокирующий, нельзя крутить в event loop напрямую.
    """
    try:
        with torch.inference_mode():
            for pcm_bytes in state.cosyvoice.inference_zero_shot_stream(
                tts_text=text,
                prompt_text=prompt_text,
                prompt_wav=str(audio_path),
                zero_shot_spk_id=spk_id,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                frequency_penalty=frequency_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                safe_pad_id=safe_pad_id,
                min_tokens_ratio=min_tokens_ratio,
                disable_text_split=disable_text_split,
                random_seed=random_seed,
                bad_speech_ids=bad_speech_ids,
                repetition_penalty=repetition_penalty,
                use_ras=use_ras,
            ):
                yield pcm_bytes
    except Exception as exc:  # noqa: BLE001
        # FastAPI StreamingResponse swallow'ит exception без логирования —
        # руками пишем в stderr чтобы видно было в docker logs.
        logger.exception("Inference failed: %s", exc)
        raise
    finally:
        # Возвращаем PyTorch CUDA caching pool системе. Делается ПОСЛЕ
        # последнего yield (генератор уже отдал клиенту всё аудио, поэтому
        # latency на следующий запрос не растёт — empty_cache занимает
        # ~30-50ms, но это уходит в idle-окно между запросами Mika.
        if EMPTY_CACHE_BETWEEN_REQUESTS and torch.cuda.is_available():
            torch.cuda.empty_cache()


@app.post("/v1/tts/zero-shot")
async def tts_zero_shot(req: TTSRequest):
    if state.cosyvoice is None:
        raise HTTPException(503, "model not loaded yet")

    if req.instruction is None:
        # Default path: cached speaker embedding (fast, ~0ms overhead).
        # TTS.Core uses this path — никакого instruction в request не шлёт.
        audio_path, prompt_text, spk_id = _ensure_speaker_registered(req.reference_id)
    else:
        # Custom-instruction path: bypass speaker cache because instruction is
        # baked into the cached embedding by add_zero_shot_spk → re-encode the
        # reference prompt each call to apply a different instruction.
        # Format mirrors inference_instruct2 from cosyvoice/cli/cosyvoice.py
        # (see run_instruct.py for reference test cases):
        #   "You are a helpful assistant. {instruction} <|endofprompt|>{transcription}"
        # Critical: SPACE before <|endofprompt|> — without it the model often
        # ignores the instruction (verified by community testing).
        audio_path, lab_path, _ = _resolve_reference(req.reference_id)
        transcription = lab_path.read_text(encoding="utf-8").strip()
        if not transcription:
            raise HTTPException(422, f"empty transcription: {lab_path}")
        user_instr = req.instruction.strip()
        # Strip any <|endofprompt|> the user pasted — we add it ourselves.
        user_instr = user_instr.replace("<|endofprompt|>", "").strip()
        if user_instr:
            prompt_text = f"You are a helpful assistant. {user_instr} <|endofprompt|>{transcription}"
        else:
            # Empty instruction = no extra directive; preserve the default
            # assistant prefix so the model sees the same structure it was
            # trained on, just without the user-supplied instruction.
            prompt_text = f"You are a helpful assistant. <|endofprompt|>{transcription}"
        spk_id = ""  # forces frontend_zero_shot to re-extract spk embedding

    # Resolve silence-fix overrides: None → use env-default.
    freq_pen = (
        req.frequency_penalty
        if req.frequency_penalty is not None
        else FREQ_PENALTY_DEFAULT
    )
    nr_ngram = (
        req.no_repeat_ngram_size
        if req.no_repeat_ngram_size is not None
        else NO_REPEAT_NGRAM_DEFAULT
    )
    safe_pad = req.safe_pad_id if req.safe_pad_id is not None else SAFE_PAD_ID_DEFAULT
    min_ratio = (
        req.min_tokens_ratio
        if req.min_tokens_ratio is not None
        else MIN_TOKENS_RATIO_DEFAULT
    )
    no_split = (
        req.disable_text_split
        if req.disable_text_split is not None
        else DISABLE_TEXT_SPLIT_DEFAULT
    )
    use_ras = req.use_ras if req.use_ras is not None else USE_RAS_DEFAULT
    rep_pen = (
        req.repetition_penalty if req.repetition_penalty is not None
        else REPETITION_PENALTY_DEFAULT
    )

    instr_log = (
        (req.instruction[:40] + "…")
        if req.instruction and len(req.instruction) > 40
        else (req.instruction or "<default>")
    )
    seed_log = req.random_seed if req.random_seed is not None else "<auto>"
    logger.info(
        "TTS [%s] T=%.2f P=%.2f K=%d fp=%.2f nrg=%d spad=%s mtr=%.2f nosplit=%s ras=%s rp=%.2f seed=%s instr=%r text=%r",
        req.reference_id,
        req.temperature,
        req.top_p,
        req.top_k,
        freq_pen,
        nr_ngram,
        safe_pad,
        min_ratio,
        no_split,
        use_ras,
        rep_pen,
        seed_log,
        instr_log,
        req.text[:80],
    )

    # 24 kHz mono PCM int16 LE — Mika.TTS.Core CosyVoiceEngine ожидает
    # ровно это (без RIFF), media_type для информативности.
    headers = {
        "X-Sample-Rate": str(state.sample_rate),
        "X-Channels": "1",
        "X-Sample-Format": "s16le",
    }
    return StreamingResponse(
        _stream_pcm(
            req.text,
            prompt_text,
            audio_path,
            spk_id,
            req.temperature,
            req.top_p,
            req.top_k,
            freq_pen,
            nr_ngram,
            safe_pad,
            min_ratio,
            no_split,
            req.random_seed,
            req.bad_speech_ids,
            rep_pen,
            use_ras,
        ),
        media_type="audio/L16; rate=24000; channels=1",
        headers=headers,
    )


# ----------------------------------------------------------------------------
# Entry point.
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "mika_server:app",
        host=HOST,
        port=PORT,
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
        # Один воркер — FastCosyVoice3 single-GPU + ~5 GB VRAM, не имеет смысла
        # держать несколько процессов на одну GPU.
        workers=1,
    )
