"""Repetition-Aware Sampling (RAS) for TRT-LLM — CORRECTED implementation.

History:
  v1 (strict, argmax + flatten range): made defect rate WORSE (0→53%, 13→47%, 7→87%)
  v2 (soft, argmax + per-token penalty): inconsistent (15-seed helped, 30-seed hurt)
  v3 (THIS): no argmax proxy. Penalize tokens with high count in BeamTokens history.

The fundamental bug in v1/v2 was using `argmax(logits)` as a proxy for "what
TRT-LLM will sample". TRT-LLM samples stochastically (T=0.7 + top_p + top_k),
so argmax is almost never the actually-chosen token. Result: penalties applied
to wrong tokens (false positives) and missed on right ones (false negatives).

v3 fix: iterate over the ACTUAL generated tokens in `token_ids[0]` (= `BeamTokens`
per TRT-LLM Executor API doc). For each unique token in the recent window, count
occurrences. If count >= max_count → penalize that token's logit. This is the
logit-space approximation of VALL-E 2 RAS Algorithm 1 (Section 4.2): instead of
"if sampled token is over-repeated, re-sample from full distribution", we
"suppress over-repeated tokens so the natural sampler picks alternatives".

API contract per tensorrt_llm.sampling_params.LogitsProcessor:
    __call__(req_id, logits, token_ids, stream_ptr, client_id) -> None
        # In-place modify `logits`. token_ids = List[List[int]] generated so far.

Registered once at FastCosyVoice3 init, activated per-request via
`logits_processor_names=['ras']` in gen_kwargs.
"""

from __future__ import annotations

from collections import Counter


class RASLogitsProcessor:
    def __init__(
        self,
        speech_token_offset: int,
        speech_token_size: int,
        win_size: int = 10,
        max_count: int = 2,
        soft_penalty: float = 3.0,
    ):
        """
        Args:
            speech_token_offset, speech_token_size: kept for compatibility / future
                range-restricted logic. Not used in v3 (we operate on full vocab
                since BeamTokens already contain only generated speech tokens).
            win_size: how many recent generated tokens to inspect (default 10,
                      matches VALL-E 2 paper).
            max_count: token must appear >= max_count times in window to trigger.
                       Default 2 — middle ground. VALL-E 2 uses 1 (any repeat
                       triggers); our soft-penalty approach is more aggressive
                       than RAS's "switch sampler", so 2 avoids over-firing on
                       natural pause tokens.
            soft_penalty: additive logit penalty per excess occurrence. Penalty
                       scales as `penalty * (count - max_count + 1)`, so a token
                       repeated 4 times in window with max_count=2 gets penalty
                       `3.0 * 3 = 9.0` (~99.99% suppression in softmax).
        """
        self.speech_lo = int(speech_token_offset)
        self.speech_hi = int(speech_token_offset) + int(speech_token_size)
        self.win_size = int(win_size)
        self.max_count = int(max_count)
        self.soft_penalty = float(soft_penalty)
        self.trigger_total: int = 0
        self.calls_total: int = 0

    def __call__(self, req_id, logits, token_ids, stream_ptr, client_id):
        self.calls_total += 1
        try:
            beam = token_ids[0] if (token_ids and isinstance(token_ids[0], (list, tuple))) else token_ids
        except Exception:
            return
        if not beam or len(beam) < self.max_count:
            return  # not enough history to detect repetition yet

        # Recent window of actually-generated tokens
        recent = beam[-self.win_size:] if len(beam) >= self.win_size else list(beam)
        counts = Counter(recent)

        # Find tokens to penalize: those with count >= max_count in window
        # (these are tokens TRT-LLM has been picking repeatedly = sign of attractor lock)
        to_penalize = [(tok, cnt) for tok, cnt in counts.items() if cnt >= self.max_count]
        if not to_penalize:
            return

        self.trigger_total += 1

        # Apply scaled penalty to each over-repeated token.
        # Penalty grows with excess count: 1 excess = base penalty, 2 excess = 2x, etc.
        # This way a token repeated 5x in window gets stronger suppression than 3x.
        if logits.dim() == 1:
            for tok, cnt in to_penalize:
                logits[tok] -= self.soft_penalty * (cnt - self.max_count + 1)
        elif logits.dim() == 2:
            for tok, cnt in to_penalize:
                logits[:, tok] -= self.soft_penalty * (cnt - self.max_count + 1)
        elif logits.dim() == 3:
            for tok, cnt in to_penalize:
                logits[:, :, tok] -= self.soft_penalty * (cnt - self.max_count + 1)
