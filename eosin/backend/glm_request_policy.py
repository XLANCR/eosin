from __future__ import annotations

import re


ADAPTIVE_FREQ_PENALTIES = [0.0, 0.10, 0.20, 0.30, 0.50]
REPETITION_PENALTY = 1.2
STOP_SEQUENCES = ["</html>"]

_CJK_RE = re.compile(r"[一-鿿぀-ゟ゠-ヿ가-힯]")
_RUNAWAY_RE = re.compile(r"([^>\s])\1{50,}")
_RUNAWAY_PAIR_RE = re.compile(r"([^>\s]{2})\1{25,}")


def glm_output_quality(html: str) -> str:
    """Classify GLM table output as good, suspicious, or bad."""
    if not html or len(html) < 50:
        return "bad"
    if _RUNAWAY_RE.search(html) or _RUNAWAY_PAIR_RE.search(html):
        return "bad"
    if "<tr" not in html and "<table" not in html:
        return "bad"

    cjk_chars = len(_CJK_RE.findall(html))
    cjk_ratio = cjk_chars / max(len(html), 1)
    if cjk_ratio > 0.3:
        return "bad"
    if cjk_ratio > 0.15:
        return "suspicious"

    row_count = len(re.findall(r"<tr[\s>]", html))
    if len(html) > 2000 and row_count < 2:
        return "suspicious"

    return "good"


def apply_glm_generation_policy(request_data: dict, *, frequency_penalty: float) -> dict:
    request_data["repetition_penalty"] = REPETITION_PENALTY
    request_data["frequency_penalty"] = frequency_penalty
    request_data["stop"] = list(STOP_SEQUENCES)
    return request_data
