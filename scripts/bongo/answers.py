"""Final-answer extraction and typing.

96.8% of the reasoning chains in ``bengali_math_cot`` and ``numina_bn`` end in a
``\\boxed{...}``; the rest fall back to the GSM8K ``####`` marker or a Bangla
``উত্তর:`` lead-in.  Extraction has to be brace-balanced because answers nest
(``\\boxed{\\frac{1}{4}}``), and it takes the *last* box because intermediate
results are sometimes boxed too.

Only some answer types can back a verifiable RL reward.  ``VERIFIABLE_TYPES``
is the subset the Phase 6 reward function should train on; everything else is
still fine as an SFT target.
"""

import re

from .normalize import to_ar_digits

# Answer types that the tiered verifier can check exactly.
VERIFIABLE_TYPES = frozenset({"numeric", "fraction", "percent", "mcq_option", "bool_bn"})

GSM_MARKER = re.compile(r"####\s*([^\n]+)")
BN_ANSWER_MARKER = re.compile(
    r"(?:উত্তর|ফলাফল)\s*(?:হচ্ছে|হলো|হল|হবে)?\s*[:ঃ]\s*([^\n]+)"
)

_BOOL_BN = {
    "সঠিক": "true", "সত্য": "true", "হ্যাঁ": "true", "হ্যাঁ।": "true",
    "ভুল": "false", "মিথ্যা": "false", "না": "false", "অসত্য": "false",
}

_NUMERIC = re.compile(r"^[-+]?\d+(?:,\d{3})*(?:\.\d+)?$")
_FRAC_TEX = re.compile(r"^\\[dt]?frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}$")
_FRAC_SLASH = re.compile(r"^(-?\d+)\s*/\s*(\d+)$")
_PERCENT = re.compile(r"^([-+]?[\d.]+)\s*\\?%$")
_MCQ = re.compile(r"^(?:\\(?:text|mathrm|textbf)\{\s*)?\(?([A-Ea-e])\)?[.)\s]*\}?$")
_SYMBOLIC = re.compile(r"\\(?:sqrt|pi|infty|sum|int|log|ln|sin|cos|tan|alpha|beta|theta)")

# "\textbf{(B)}\ 2", "\text{A: }0.8", "C: 825" — an option letter carrying the
# value it stands for. The letter is the answer; the payload is accepted too,
# because a model may reasonably state either.
_MCQ_PAYLOAD = re.compile(
    r"^\\?(?:text|textbf|mathrm)?\s*\{?\s*\(?([A-Ea-e])\)?\s*[:.)]\s*\}?\s*(.+)$", re.S
)
_CURRENCY = re.compile(r"^\\?[$₹৳]\s*([-+]?[\d,]+(?:\.\d+)?)$")
_UNIT_SUFFIX = re.compile(
    r"^([-+]?[\d,]+(?:\.\d+)?)\s*(?:টাকা|টি|টা|জন|মিটার|কিমি|কিলোমিটার|সেমি|"
    r"সেন্টিমিটার|ঘন্টা|ঘণ্টা|মিনিট|সেকেন্ড|দিন|বছর|মাস|কেজি|গ্রাম|লিটার|ডিগ্রি|"
    r"বর্গমিটার|cm|mm|km|kg|g|ml|hr|min|sec|units?|dollars?)\.?$"
)


def extract_boxed(text: str):
    """Return the payload of the last brace-balanced ``\\boxed{...}``, or None."""
    if not text or "\\boxed" not in text:
        return None
    out, i = None, 0
    while True:
        start = text.find("\\boxed", i)
        if start < 0:
            break
        brace = text.find("{", start)
        if brace < 0:
            break
        depth, pos = 0, brace
        while pos < len(text):
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
                if depth == 0:
                    break
            pos += 1
        if depth != 0:  # unbalanced; give up on this occurrence
            break
        out = text[brace + 1 : pos]
        i = pos + 1
    return out


def extract_answer(*fields):
    """Try each field in order, returning ``(raw_answer, method)`` or (None, None).

    Fields are typically ``(solution, answer)``.  A field that is already a short
    bare answer is accepted as-is.
    """
    for text in fields:
        text = (text or "").strip()
        if not text:
            continue
        boxed = extract_boxed(text)
        if boxed is not None and boxed.strip():
            return boxed.strip(), "boxed"
        m = GSM_MARKER.search(text)
        if m and m.group(1).strip():
            return m.group(1).strip(), "gsm_marker"
        m = BN_ANSWER_MARKER.search(text)
        if m and m.group(1).strip():
            return m.group(1).strip(), "bn_marker"
    # Fall back to a field that is itself already a bare answer.
    for text in fields:
        text = (text or "").strip()
        if text and len(text) <= 40 and "\n" not in text:
            return text, "bare"
    return None, None


def _clean(answer: str) -> str:
    """Strip decoration that never carries meaning: $, \\!, \\,, trailing punctuation."""
    a = answer.strip()
    a = re.sub(r"^\$+|\$+$", "", a).strip()
    a = re.sub(r"\\[!,;:]", "", a)
    a = re.sub(r"\\(?:left|right|displaystyle|textstyle)\b", "", a)
    a = a.replace("\\ ", " ").strip()
    # Trailing comma only when it is not a thousands separator ("10," but not "5,050").
    a = re.sub(r"(?<=\d),\s*$", "", a)
    a = re.sub(r"[.।]\s*$", "", a).strip()
    return a


def mcq_payload(answer: str):
    """For "\\textbf{(B)}\\ 2" return ("B", "2"); otherwise (None, None)."""
    if answer is None:
        return None, None
    a = _clean(answer)
    if _MCQ.match(a):  # bare letter, no payload
        return None, None
    m = _MCQ_PAYLOAD.match(a)
    if not m:
        return None, None
    payload = m.group(2).strip().rstrip("}").strip()
    return m.group(1).upper(), (payload or None)


def _strip_units(compact: str) -> str:
    """Reduce "\\$50" or "60 টাকা" to "50" / "60"; returns input unchanged if neither."""
    m = _CURRENCY.match(compact)
    if m:
        return m.group(1)
    m = _UNIT_SUFFIX.match(compact)
    if m:
        return m.group(1)
    return compact


def classify(answer: str) -> str:
    """Bucket a raw extracted answer into an ``answer_type``."""
    if answer is None:
        return "missing"
    a = to_ar_digits(_clean(answer))
    if not a:
        return "missing"

    inner = re.sub(r"^\\(?:text|mathrm|textbf|textit)\{(.*)\}$", r"\1", a).strip()
    if inner in _BOOL_BN:
        return "bool_bn"
    if _MCQ.match(a):
        return "mcq_option"
    if mcq_payload(answer)[0]:
        return "mcq_option"

    compact = _strip_units(a.replace(" ", ""))
    if _NUMERIC.match(compact):
        return "numeric"
    if _PERCENT.match(compact):
        return "percent"
    if _FRAC_TEX.match(compact) or _FRAC_SLASH.match(compact):
        return "fraction"
    if _SYMBOLIC.search(compact):
        return "symbolic"
    if re.search(r"[a-zA-Z\\]", compact):
        return "expression"
    return "other"


def canonical(answer: str, answer_type: str = None):
    """Comparable canonical form, or None when the type is not verifiable.

    numeric/percent -> float, fraction -> (num, den) reduced, mcq -> "A",
    bool_bn -> "true"/"false".
    """
    if answer is None:
        return None
    answer_type = answer_type or classify(answer)
    a = to_ar_digits(_clean(answer))
    compact = _strip_units(a.replace(" ", ""))

    if answer_type == "numeric":
        try:
            return float(compact.replace(",", ""))
        except ValueError:
            return None
    if answer_type == "percent":
        m = _PERCENT.match(compact)
        return float(m.group(1)) if m else None
    if answer_type == "fraction":
        m = _FRAC_TEX.match(compact) or _FRAC_SLASH.match(compact)
        if not m:
            return None
        try:
            from fractions import Fraction

            f = Fraction(m.group(1)) / Fraction(m.group(2))
            return (f.numerator, f.denominator)
        except (ValueError, ZeroDivisionError):
            return None
    if answer_type == "mcq_option":
        m = _MCQ.match(a)
        if m:
            return m.group(1).upper()
        return mcq_payload(answer)[0]
    if answer_type == "bool_bn":
        inner = re.sub(r"^\\(?:text|mathrm|textbf|textit)\{(.*)\}$", r"\1", a).strip()
        return _BOOL_BN.get(inner)
    return None
