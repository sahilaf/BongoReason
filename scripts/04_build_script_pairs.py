"""Stage 4 — build genuine dual-script pairs (MANIFEST B3).

The v0.3 dataset stored a single ``*_digits`` field that only ever converted
*towards* Bengali numerals, so it held 0% Arabic numerals and was byte-identical
to the original wherever the source was already in Bengali digits — no contrast
to learn from, and no way to state the consistency reward.

This stage emits explicit both-direction pairs and keeps only the ones that are
non-degenerate, i.e. the question actually contains a digit, so the two variants
differ.  Digit-free word problems are excluded rather than padded in: a pair
whose halves are identical would score a perfect consistency reward for free and
inflate the metric.

Usage:  python scripts/04_build_script_pairs.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bongo import DATASET
from bongo.normalize import has_ar_digits, has_bn_digits, to_ar_digits, to_bn_digits

IN_PATH = DATASET / "verified" / "bongo_reason_v0.4.json"
OUT_DIR = DATASET / "script_pairs"
REPORT_PATH = DATASET / "metadata" / "stage04_script_pairs_report.json"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    records = json.loads(IN_PATH.read_text(encoding="utf-8"))
    print(f"loaded {len(records):,} records\n")

    pairs, degenerate = [], 0
    for r in records:
        question = r["question_bn"]
        if not (has_ar_digits(question) or has_bn_digits(question)):
            degenerate += 1
            continue

        ar_q, bn_q = to_ar_digits(question), to_bn_digits(question)
        if ar_q == bn_q:  # digits present but conversion is a no-op
            degenerate += 1
            continue

        pair = {
            "id": r["id"],
            "source": r["source"],
            "pool": r["pool"],
            "answer_type": r["answer_type"],
            "verifiable": r["verifiable"],
            "question_ar_digits": ar_q,
            "question_bn_digits": bn_q,
            "final_answer_ar_digits": to_ar_digits(r["final_answer"]),
            "final_answer_bn_digits": to_bn_digits(r["final_answer"]),
        }
        if r["solution_bn"]:
            pair["solution_ar_digits"] = to_ar_digits(r["solution_bn"])
            pair["solution_bn_digits"] = to_bn_digits(r["solution_bn"])
        pairs.append(pair)

    print("=" * 78)
    print("DUAL-SCRIPT PAIRS")
    print("=" * 78)
    print(f"  non-degenerate pairs: {len(pairs):,}")
    print(f"  excluded (no digits in question, variants identical): {degenerate:,}")

    by_source = Counter(p["source"] for p in pairs)
    by_pool = Counter(p["pool"] for p in pairs)
    print(f"\n  by pool: {dict(by_pool)}")
    print("  by source:")
    source_totals = Counter(r["source"] for r in records)
    for s, c in by_source.most_common():
        print(f"    {s:<24}{c:>7,} of {source_totals[s]:,} ({100*c/source_totals[s]:.0f}%)")

    # Which script did each source originally use? Useful for the paper: it says
    # which direction the model has actually seen during pretraining.
    print("\n  original numeral script of the question:")
    orig = Counter()
    for r in records:
        q = r["question_bn"]
        if has_ar_digits(q) and has_bn_digits(q):
            orig[(r["source"], "mixed")] += 1
        elif has_ar_digits(q):
            orig[(r["source"], "arabic")] += 1
        elif has_bn_digits(q):
            orig[(r["source"], "bengali")] += 1
        else:
            orig[(r["source"], "none")] += 1
    for source in sorted(source_totals):
        row = {k[1]: v for k, v in orig.items() if k[0] == source}
        total = source_totals[source]
        parts = "  ".join(
            f"{k}:{100*v/total:>3.0f}%" for k, v in sorted(row.items(), key=lambda kv: -kv[1])
        )
        print(f"    {source:<24}{parts}")

    with_solution = sum(1 for p in pairs if "solution_ar_digits" in p)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "script_pairs_v0.4.json"
    out_path.write_text(json.dumps(pairs, ensure_ascii=False, indent=1), encoding="utf-8")

    REPORT_PATH.write_text(
        json.dumps(
            {
                "stage": "04_build_script_pairs",
                "input_records": len(records),
                "pairs": len(pairs),
                "excluded_degenerate": degenerate,
                "pairs_with_solution": with_solution,
                "by_pool": dict(by_pool),
                "by_source": dict(by_source),
                "original_script": {f"{k[0]}/{k[1]}": v for k, v in orig.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  pairs carrying a solution: {with_solution:,}")
    print(f"\nwrote {out_path.relative_to(DATASET.parent)}")
    print(f"wrote {REPORT_PATH.relative_to(DATASET.parent)}")


if __name__ == "__main__":
    main()
