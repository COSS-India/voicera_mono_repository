"""Streaming-ASR metric definitions.

These are written out rather than imported because most of them are argued about, and the
argument matters more than the arithmetic.
"""
from __future__ import annotations

import unicodedata


# ---------------------------------------------------------------------------------------
# accuracy / sanity
# ---------------------------------------------------------------------------------------
def normalize(s: str) -> str:
    """NFC + whitespace collapse. Nothing else.

    Deliberately no case folding, punctuation stripping or digit normalisation: this corpus
    spans 11 scripts, and 'helpful' normalisation written against Latin text silently mangles
    Perso-Arabic and Indic combining marks.
    """
    return " ".join(unicodedata.normalize("NFC", s).split())


def _levenshtein(a: list, b: list) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float | None:
    """Character error rate against the model's OWN offline transcript.

    Accuracy is not what this campaign measures. This is a SANITY metric: the question is
    whether streaming output degrades relative to what the same weights produce offline, not
    whether the model is right about the audio. Scoring against the model itself removes the
    dataset's own labelling noise from the comparison.
    """
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return None
    return round(_levenshtein(list(r), list(h)) / len(r), 4)


# ---------------------------------------------------------------------------------------
# smoothness
# ---------------------------------------------------------------------------------------
def split_gaps(partials: list[dict]) -> tuple[list[float], list[float]]:
    """Inter-commit gaps, split into within-turn and rotation-spanning.

    Returns (steady, boundary). A gap is `boundary` when the two commits bracketing it carry
    different turn indices -- the decoder was restarted between them, which is the visible
    pause. Pooling the two produces a p99 that describes neither, so they are kept apart.

    EVERY commit counts, `turn_final` included: it updates the transcript on screen exactly as
    a partial does, and excluding it measures a gap the user never experienced.
    """
    body = list(partials)
    steady, boundary = [], []
    for i in range(1, len(body)):
        gap = body[i]["t_ms"] - body[i - 1]["t_ms"]
        prev_turn, turn = body[i - 1].get("turn"), body[i].get("turn")
        if prev_turn is not None and turn is not None and turn != prev_turn:
            boundary.append(gap)
        else:
            steady.append(gap)
    return steady, boundary


def leading_silence_ms(wav, thresh: float = 0.01, win: int = 1600) -> float:
    """Milliseconds of near-silence before speech starts.

    TTFP is honestly measured from the first audio byte -- that is when the client started
    sending. But a clip opening with two seconds of silence then reports a two-second worse
    TTFP than the same clip trimmed, and that difference is the corpus, not the server.
    """
    import numpy as np
    if len(wav) < win:
        return 0.0
    n = len(wav) // win
    fr = np.abs(wav[:n * win].reshape(n, win)).max(axis=1)
    loud = np.nonzero(fr > thresh)[0]
    return round(float(loud[0] * win) / 16000 * 1000, 1) if len(loud) else 0.0


# ---------------------------------------------------------------------------------------
# stability
# ---------------------------------------------------------------------------------------
def normalized_erasure(fulls: list[str]) -> float:
    """NE = (1/J) * sum(|o_{i-1}| - |LCP(o_i, o_{i-1})|)

    How much previously-shown text each update retracts. **AlignAtt is append-only, so this
    should be exactly 0.** It is measured rather than assumed: a nonzero value is a real bug,
    not a tuning result.
    """
    if len(fulls) < 2:
        return 0.0
    total = 0
    for prev, cur in zip(fulls, fulls[1:]):
        lcp = 0
        for a, b in zip(prev, cur):
            if a != b:
                break
            lcp += 1
        total += len(prev) - lcp
    return round(total / (len(fulls) - 1), 4)


def flicker(fulls: list[str]) -> dict:
    """UPWR / UPSR: how often an update rewrites, and how much when it does.

    Reported as a pair because a system can flicker rarely but catastrophically, and a single
    averaged number hides exactly that.
    """
    if len(fulls) < 2:
        return {"upsr": 0.0, "upwr": 0.0}
    rewrites, chars = 0, 0
    for prev, cur in zip(fulls, fulls[1:]):
        lcp = 0
        for a, b in zip(prev, cur):
            if a != b:
                break
            lcp += 1
        if lcp < len(prev):
            rewrites += 1
            chars += len(prev) - lcp
    n = len(fulls) - 1
    return {
        "upsr": round(rewrites / n, 4),                       # frequency
        "upwr": round(chars / max(1, rewrites), 2),           # magnitude when it happens
    }


# ---------------------------------------------------------------------------------------
# lag
# ---------------------------------------------------------------------------------------
def laal(emissions: list[tuple[float, int]], audio_s: float, n_ref_tokens: int) -> float | None:
    """Length-Adaptive Average Lag, in seconds.

        LAAL = (1/tau') * sum_i (d_i - d_i*),  d_i* = (i-1) * |X| / max(|Y|, |Y*|)

    **LAAL, never AL.** AL normalises by |Y*| (the reference length) alone, so a system that
    over-generates accumulates negative terms and scores BETTER. It literally rewards
    hallucination -- which is this model family's exact failure mode (6 s of silence producing
    one token 45 times). LAAL's max(|Y|, |Y*|) denominator removes that incentive.

    `emissions` is [(wall_clock_seconds_since_audio_start, n_tokens_emitted), ...]. Because the
    times are wall clock this is the COMPUTATION-AWARE variant: it includes our own compute,
    which is the number a caller actually experiences.
    """
    if not emissions or audio_s <= 0:
        return None
    n_hyp = sum(n for _, n in emissions)
    denom = max(n_hyp, n_ref_tokens)
    if denom <= 0:
        return None
    per_token = audio_s / denom

    total, count, idx = 0.0, 0, 0
    for t, n in emissions:
        for _ in range(n):
            idx += 1
            total += t - (idx - 1) * per_token
            count += 1
            if idx >= denom:      # tau': stop at the first index reaching the denominator
                return round(total / count, 4)
    return round(total / count, 4) if count else None


def percentile(xs: list, q: float):
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(q * len(s)))], 2)


def bootstrap_ci(xs: list, q: float = 0.5, n: int = 1000, seed: int = 0) -> tuple | None:
    """Percentile bootstrap CI. Two configs whose intervals overlap are NOT different.

    Deterministic seed so a reported interval can be reproduced exactly.
    """
    if len(xs) < 3:
        return None
    import random

    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        sample = [xs[rng.randrange(len(xs))] for _ in range(len(xs))]
        stats.append(percentile(sample, q))
    stats.sort()
    return (round(stats[int(0.025 * n)], 2), round(stats[int(0.975 * n)], 2))
