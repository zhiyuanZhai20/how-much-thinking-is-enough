"""Extract \\boxed{} answers from model output and compare for equality."""
from __future__ import annotations
import re
from fractions import Fraction


_BOXED_RE = re.compile(r"\\boxed\s*{")


def extract_boxed(text: str) -> str | None:
    """Find the LAST \\boxed{...} in text and return inner content (balanced braces)."""
    if not text:
        return None
    # find all starts, take the last
    starts = [m.end() for m in _BOXED_RE.finditer(text)]
    if not starts:
        return None
    start = starts[-1]
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1
    return None


def extract_answer(text: str) -> str | None:
    """Best-effort: \\boxed first, else last number in text."""
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text or "")
    return nums[-1] if nums else None


def _normalize_str(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    s = s.replace(" ", "").replace("\\,", "").replace(",", "")
    s = s.replace("\\!", "").replace("$", "")
    s = s.rstrip(".")
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    return s


def _try_number(s: str):
    s = s.replace("\\frac", "")
    s = s.replace("{", "").replace("}", "")
    if "/" in s:
        try:
            num, den = s.split("/", 1)
            return Fraction(int(num), int(den))
        except Exception:
            pass
    try:
        return Fraction(s).limit_denominator(10**9)
    except Exception:
        try:
            return float(s)
        except Exception:
            return None


def answers_equal(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    sa, sb = _normalize_str(a), _normalize_str(b)
    if sa == sb:
        return True
    na, nb = _try_number(sa), _try_number(sb)
    if na is not None and nb is not None:
        try:
            return abs(float(na) - float(nb)) < 1e-6
        except Exception:
            return na == nb
    return False
