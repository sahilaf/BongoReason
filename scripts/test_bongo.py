"""Tests for the extraction/verification library.  Run: python scripts/test_bongo.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bongo.answers import (
    canonical,
    classify,
    extract_answer,
    extract_boxed,
    mcq_payload,
)
from bongo.normalize import (
    normalize_for_matching,
    to_ar_digits,
    to_bn_digits,
    is_bangla_dominant,
)
from bongo.verify import verify

FAILURES = []


def check(name, got, want):
    if got != want:
        FAILURES.append(f"{name}: got {got!r}, want {want!r}")


def check_true(name, got):
    if not got:
        FAILURES.append(f"{name}: expected truthy, got {got!r}")


def check_false(name, got):
    if got:
        FAILURES.append(f"{name}: expected falsy, got {got!r}")


# ---------- boxed extraction ----------
check("nested braces", extract_boxed(r"\[ P(Y) = \boxed{\frac{1}{4}} \]"), r"\frac{1}{4}")
check("plain", extract_boxed(r"so $\boxed{243}$।"), "243")
check("takes last box", extract_boxed(r"\boxed{12} then \boxed{7}"), "7")
check("deep nesting", extract_boxed(r"\boxed{\text{A: } \frac{a}{\frac{b}{c}}}"),
      r"\text{A: } \frac{a}{\frac{b}{c}}")
check("absent", extract_boxed("no box here"), None)
check("unbalanced", extract_boxed(r"\boxed{12"), None)

# ---------- fallback extraction ----------
check("gsm marker", extract_answer("...\n#### ৩৬"), ("৩৬", "gsm_marker"))
check("bangla marker", extract_answer("অতএব উত্তর: 42"), ("42", "bn_marker"))
check("bare field", extract_answer("", "18"), ("18", "bare"))
check("boxed wins over bare", extract_answer(r"x = \boxed{5}", "9"), ("5", "boxed"))
check("nothing", extract_answer("", ""), (None, None))

# ---------- classification ----------
for raw, want in [
    ("91", "numeric"), ("-12", "numeric"), ("5,050", "numeric"), ("3.14", "numeric"),
    (r"\frac{7}{16}", "fraction"), ("3/4", "fraction"),
    (r"10\%", "percent"),
    (r"\text{C}", "mcq_option"), ("A", "mcq_option"), (r"\textbf{(B)}", "mcq_option"),
    (r"\text{সঠিক}", "bool_bn"), (r"\text{ভুল}", "bool_bn"),
    (r"\frac{9\sqrt{3}}{2}", "symbolic"), (r"-\frac{\pi}{3}", "symbolic"),
    ("2x - 2y + z - 1 = 0", "expression"),
    ("(2, 3)", "other"), ("4:3", "other"),
]:
    check(f"classify {raw!r}", classify(raw), want)

check("bengali numerals classify as numeric", classify("৩৬"), "numeric")
check("dollar-wrapped", classify("$42$"), "numeric")

# option letters that carry their value
for raw in [r"\textbf{(B)}\ 2", r"\text{A: }0.8", "C: 825", r"\textbf{B) 47.73\%}"]:
    check(f"mcq payload {raw!r}", classify(raw), "mcq_option")
check("mcq payload split", mcq_payload(r"\textbf{(B)}\ 2"), ("B", "2"))
check("bare letter has no payload", mcq_payload(r"\text{C}"), (None, None))
check("canon mcq from payload form", canonical(r"\textbf{(C)}\ 70"), "C")

# currency and units
check("currency", classify(r"\$50"), "numeric")
check("canon currency", canonical(r"\$50"), 50.0)
check("bangla unit", classify("৬০ টাকা"), "numeric")
check("canon bangla unit", canonical("৬০ টাকা"), 60.0)
check("metric unit", classify("22cm"), "numeric")
check("trailing comma stripped", canonical("10,"), 10.0)
check("thousands separator survives", canonical("5,050"), 5050.0)

# ---------- canonical ----------
check("canon numeric", canonical("5,050"), 5050.0)
check("canon bengali numeric", canonical("৩৬"), 36.0)
check("canon fraction reduces", canonical(r"\frac{2}{8}"), (1, 4))
check("canon mcq", canonical(r"\text{c}"), "C")
check("canon bool", canonical(r"\text{সঠিক}"), "true")
check("canon symbolic is None", canonical(r"\sqrt{2}"), None)

# ---------- numeral conversion ----------
check("to bn", to_bn_digits("42 apples"), "৪২ apples")
check("to ar", to_ar_digits("৪২টি"), "42টি")
check("roundtrip", to_ar_digits(to_bn_digits("2026")), "2026")
check("match form collapses script",
      normalize_for_matching("৪২ টি আম!"), normalize_for_matching("42টি আম"))

# ---------- language detection ----------
check_true("english chain detected",
           not is_bangla_dominant("To solve this problem, let's break it down step by step"))
check_true("bangla chain detected", is_bangla_dominant("অতএব সঠিক উত্তর হল ২৪৩ টি।"))
check_true("math-heavy bangla still bangla",
           is_bangla_dominant(r"যেহেতু $f(x) = \frac{1}{2}$ তাই মান হবে ২।"))

# ---------- verification ----------
check_true("exact numeric", verify(r"উত্তর \boxed{36}", "36"))
check_true("bengali pred vs arabic gold", verify(r"\boxed{৩৬}", "36"))
check_true("arabic pred vs bengali gold", verify(r"\boxed{36}", "৩৬"))
check_true("fraction equality", verify(r"\boxed{\frac{2}{8}}", r"\frac{1}{4}"))
check_true("cross type frac vs decimal", verify(r"\boxed{0.25}", r"\frac{1}{4}"))
check_true("mcq case insensitive", verify(r"\boxed{\text{c}}", r"\text{C}"))
check_true("bool", verify(r"\boxed{\text{সঠিক}}", r"\text{সত্য}"))
check_false("wrong numeric", verify(r"\boxed{35}", "36"))
check_false("no answer in prediction", verify("I am not sure", "36"))
check_false("symbolic is unverifiable", verify(r"\boxed{\sqrt{2}}", r"\sqrt{2}"))
check("unverifiable reason", verify(r"\boxed{\sqrt 2}", r"\sqrt{2}").reason,
      "unverifiable_type:symbolic")
check_true("float tolerance", verify(r"\boxed{0.1}", "0.10000000001"))
check_true("mcq letter vs payload gold", verify(r"\boxed{B}", r"\textbf{(B)}\ 2"))
check_true("mcq payload value accepted", verify(r"\boxed{2}", r"\textbf{(B)}\ 2"))
check_false("mcq wrong letter and value", verify(r"\boxed{C}", r"\textbf{(B)}\ 2"))
check_true("currency pred vs plain gold", verify(r"\boxed{\$50}", "50"))
check_true("bengali unit answer", verify(r"\boxed{৬০ টাকা}", "60"))

# ---------- report ----------
if FAILURES:
    print(f"FAILED ({len(FAILURES)}):")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print("all tests passed")
