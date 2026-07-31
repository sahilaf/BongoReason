"""Text and numeral normalization for Bangla math data.

The dual-script work needs numeral conversion in *both* directions, and the
converters must not touch digits that are part of LaTeX control sequences or
identifiers (``\\frac12`` is fine to convert, ``H2O`` and ``\\pi_1`` are not
digits we want to rewrite in running prose but inside math they are).  The
approach here is deliberately conservative: convert every decimal digit, but
leave the surrounding markup alone.
"""

import re
import unicodedata

BN_DIGITS = "০১২৩৪৫৬৭৮৯"
AR_DIGITS = "0123456789"

_TO_BN = str.maketrans(AR_DIGITS, BN_DIGITS)
_TO_AR = str.maketrans(BN_DIGITS, AR_DIGITS)

BENGALI_RANGE = re.compile(r"[ঀ-৿]")
LATIN_RANGE = re.compile(r"[A-Za-z]")
CJK_RANGE = re.compile(r"[　-〿一-鿿＀-￯]")

_WS = re.compile(r"[ \t ​‌‍]+")
_NL = re.compile(r"\n{3,}")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([।,;:?!)])")
_MATH_SPAN = re.compile(r"(\$\$.*?\$\$|\$[^$]*\$|\\\[.*?\\\]|\\\(.*?\\\))", re.S)


def to_bn_digits(text: str) -> str:
    """Rewrite Arabic-Indic (ASCII) digits as Bengali digits."""
    return text.translate(_TO_BN)


def to_ar_digits(text: str) -> str:
    """Rewrite Bengali digits as ASCII digits."""
    return text.translate(_TO_AR)


def has_bn_digits(text: str) -> bool:
    return any(c in BN_DIGITS for c in text)


def has_ar_digits(text: str) -> bool:
    return any(c in AR_DIGITS for c in text)


def strip_math(text: str) -> str:
    """Remove LaTeX math spans and control sequences, for language detection."""
    text = _MATH_SPAN.sub(" ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    return text


def script_profile(text: str) -> dict:
    """Letter counts by script, ignoring math. Used to detect English chains."""
    plain = strip_math(text)
    bengali = len(BENGALI_RANGE.findall(plain))
    latin = len(LATIN_RANGE.findall(plain))
    total = bengali + latin
    return {
        "bengali": bengali,
        "latin": latin,
        "bengali_ratio": bengali / total if total else 0.0,
        "has_cjk": bool(CJK_RANGE.search(text)),
    }


def is_bangla_dominant(text: str, threshold: float = 0.6) -> bool:
    return script_profile(text)["bengali_ratio"] > threshold


def normalize_text(text: str) -> str:
    """Canonical form for storage: NFC, collapsed whitespace, tidy punctuation.

    Numerals are left exactly as they are — script normalization is a separate,
    explicit step so that the original numeral script stays recoverable.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _NL.sub("\n\n", text)
    return text.strip()


def normalize_for_matching(text: str) -> str:
    """Aggressive form used for dedup and contamination checks only.

    Collapses numeral script, drops punctuation and whitespace, lowercases.
    Two problems that differ only in numeral script must collide here.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text).lower()
    text = to_ar_digits(text)
    text = re.sub(r"[^\wঀ-৿]+", "", text)
    return text


def shingles(text: str, k: int = 5) -> set:
    """Character k-shingles over the matching form, for near-duplicate MinHash."""
    s = normalize_for_matching(text)
    if len(s) < k:
        return {s} if s else set()
    return {s[i : i + k] for i in range(len(s) - k + 1)}
