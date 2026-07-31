"""Tests for the dual-script agreement metrics. Run: python scripts/test_eval_metrics.py

The agreement rate is the paper's headline number, so it is worth pinning with a
fixture whose answer can be counted by hand. Note that agreement is about the two
scripts producing the *same* answer, not the *right* one — two identically wrong
answers agree.
"""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

spec = importlib.util.spec_from_file_location("run_eval", HERE / "run_eval.py")
run_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_eval)

# Ten problems, gold answer 10, hand-countable:
#   0-3  both scripts answer 10  -> agree, both correct        (4)
#   4-5  ar 10, bn 99            -> disagree, correctness flip (2)
#   6-7  both answer 77          -> agree, both wrong          (2)
#   8-9  ar 55, bn 66            -> disagree, both wrong       (2)
#
# agreement = (4 + 2) / 10 = 0.6   <- same-wrong-answer pairs still agree
# acc(ar)   = (4 + 2) / 10 = 0.6
# acc(bn)   =  4      / 10 = 0.4
CASES = [
    (("10", True), ("10", True)),
    (("10", True), ("10", True)),
    (("10", True), ("10", True)),
    (("10", True), ("10", True)),
    (("10", True), ("99", False)),
    (("10", True), ("99", False)),
    (("77", False), ("77", False)),
    (("77", False), ("77", False)),
    (("55", False), ("66", False)),
    (("55", False), ("66", False)),
]

EXPECTED = {
    "n_paired": 10,
    "accuracy_ar": 0.6,
    "accuracy_bn": 0.4,
    "accuracy_gap": 0.2,
    "answer_agreement": 0.6,
    "both_correct": 0.4,
    "correctness_flip": 0.2,
}


def write_fixture(path, extra=""):
    rows = []
    for i, (ar, bn) in enumerate(CASES):
        for script, (pred, ok) in (("ar", ar), ("bn", bn)):
            rows.append({
                "eval_id": f"t_{i}", "benchmark": "bn_mgsm", "script": script,
                "gold_answer": "10", "answer_type": "numeric",
                "predicted_raw": pred, "predicted_canonical": repr(float(pred)),
                "extraction_method": "boxed", "correct": ok,
                "reason": "numeric_compare", "output_chars": 100, "output": None,
            })
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + extra,
        encoding="utf-8",
    )


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    failures = []
    tmp = Path(tempfile.mkdtemp())

    path = tmp / "bn_mgsm.jsonl"
    write_fixture(path)

    # Silence the report table; we only want the returned numbers.
    import io, contextlib

    with contextlib.redirect_stdout(io.StringIO()):
        summary = run_eval.report([path], ["bn_mgsm"])
        # A killed Colab run leaves a half-written final line.
        truncated = tmp / "trunc.jsonl"
        write_fixture(truncated, extra='\n{"eval_id": "cut')
        summary_trunc = run_eval.report([truncated], ["bn_mgsm"])

    got = summary["bn_mgsm"]
    for key, want in EXPECTED.items():
        if abs(got[key] - want) > 1e-9:
            failures.append(f"{key}: got {got[key]!r}, want {want!r}")

    if summary_trunc["bn_mgsm"]["n_paired"] != 10:
        failures.append("truncated trailing line broke parsing")

    done = run_eval.load_done(path)
    if len(done) != 20 or ("t_0", "ar") not in done:
        failures.append(f"load_done returned {len(done)} items, expected 20")

    # An unpaired item (one script only) must not enter the metric.
    with path.open("a", encoding="utf-8") as f:
        f.write("\n" + json.dumps({
            "eval_id": "solo", "benchmark": "bn_mgsm", "script": "ar",
            "gold_answer": "10", "answer_type": "numeric", "predicted_raw": "10",
            "predicted_canonical": "10.0", "extraction_method": "boxed",
            "correct": True, "reason": "numeric_compare", "output_chars": 10,
            "output": None,
        }, ensure_ascii=False))
    with contextlib.redirect_stdout(io.StringIO()):
        after = run_eval.report([path], ["bn_mgsm"])["bn_mgsm"]
    if after["n_paired"] != 10:
        failures.append(f"unpaired item leaked into metric: n={after['n_paired']}")

    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print("  -", f)
        return 1
    print("all eval metric tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
