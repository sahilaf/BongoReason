"""Tiered answer verifier — the reward signal for Phase 6 GRPO.

Gold answers in this dataset are almost entirely Arabic-numeral (7,492 Arabic
vs 4 Bengali among extracted numerics), but a Bangla-native model should be free
to answer in Bengali numerals.  Every comparison therefore collapses numeral
script first; skipping that would score correct Bengali-numeral generations as
wrong and silently invert the dual-script reward.
"""

from dataclasses import dataclass

from .answers import (
    VERIFIABLE_TYPES,
    canonical,
    classify,
    extract_answer,
    mcq_payload,
)

REL_TOL = 1e-6
ABS_TOL = 1e-9


@dataclass(frozen=True)
class Result:
    correct: bool
    reason: str
    gold_canonical: object = None
    pred_canonical: object = None

    def __bool__(self):
        return self.correct


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(ABS_TOL, REL_TOL * max(abs(a), abs(b)))


def verify(prediction: str, gold: str, answer_type: str = None) -> Result:
    """Check a predicted answer against gold.

    ``prediction`` may be a full generation; the final answer is extracted from
    it the same way it was extracted from the training data.
    """
    answer_type = answer_type or classify(gold)
    if answer_type not in VERIFIABLE_TYPES:
        return Result(False, f"unverifiable_type:{answer_type}")

    gold_c = canonical(gold, answer_type)
    if gold_c is None:
        return Result(False, "gold_uncanonicalizable")

    pred_raw, _ = extract_answer(prediction)
    if pred_raw is None:
        return Result(False, "no_answer_in_prediction", gold_c, None)

    # Compare in the gold's type first, then permit numeric/fraction/percent
    # cross-type equality (1/4 vs 0.25 vs 25% are the same answer).
    pred_c = canonical(pred_raw, answer_type)
    if pred_c is None:
        pred_type = classify(pred_raw)
        if {answer_type, pred_type} <= {"numeric", "fraction", "percent"}:
            gold_f, pred_f = _as_float(gold, answer_type), _as_float(pred_raw, pred_type)
            if gold_f is not None and pred_f is not None:
                ok = _close(gold_f, pred_f)
                return Result(ok, "cross_type_numeric", gold_f, pred_f)
        if answer_type == "mcq_option" and _payload_matches(gold, pred_raw):
            return Result(True, "mcq_payload_compare", gold_c, pred_raw)
        return Result(False, f"pred_uncanonicalizable:{pred_type}", gold_c, None)

    if answer_type in ("numeric", "percent"):
        return Result(_close(gold_c, pred_c), "numeric_compare", gold_c, pred_c)
    if answer_type == "fraction":
        return Result(gold_c == pred_c, "fraction_compare", gold_c, pred_c)
    if answer_type == "mcq_option" and gold_c != pred_c and _payload_matches(gold, pred_raw):
        return Result(True, "mcq_payload_compare", gold_c, pred_raw)
    return Result(gold_c == pred_c, f"{answer_type}_compare", gold_c, pred_c)


def _payload_matches(gold: str, pred_raw: str) -> bool:
    """Accept "2" against a gold of "\\textbf{(B)}\\ 2" — the option's value."""
    _, payload = mcq_payload(gold)
    if payload is None:
        return False
    payload_type = classify(payload)
    if payload_type not in VERIFIABLE_TYPES:
        return False
    a, b = canonical(payload, payload_type), canonical(pred_raw, payload_type)
    if a is None or b is None:
        return False
    if isinstance(a, float) and isinstance(b, float):
        return _close(a, b)
    return a == b


def _as_float(answer: str, answer_type: str):
    c = canonical(answer, answer_type)
    if c is None:
        return None
    if isinstance(c, tuple):  # fraction
        return c[0] / c[1]
    if isinstance(c, (int, float)):
        return float(c) / 100 if answer_type == "percent" else float(c)
    return None


def is_verifiable(gold: str, answer_type: str = None) -> bool:
    """True when this gold answer can back an RL reward."""
    answer_type = answer_type or classify(gold)
    return answer_type in VERIFIABLE_TYPES and canonical(gold, answer_type) is not None
