"""Stage 6 — build dual-script evaluation sets.

Emits every benchmark problem twice, once with all numerals in Arabic digits and
once in Bengali digits, so that cross-script *agreement* can be measured. No
existing Bangla math benchmark does this: BanglaMATH is ~95% Arabic-digit and
GSM-Plus-BN is Bengali-digit, so comparing across them confounds numeral script
with problem difficulty. Comparing a problem against itself does not.

Pairs whose two variants are identical (the question contains no digits) are
kept but marked ``variants_differ: false``. They must be excluded from the
agreement metric — a pair that cannot disagree would score a free match and
inflate the number.

Answers are typed with the same classifier used on training data, so the
agreement metric is computed only over problems the verifier can actually grade.

Usage:  python scripts/06_build_eval_sets.py
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bongo import DATASET
from bongo.answers import classify
from bongo.normalize import normalize_text, to_ar_digits, to_bn_digits
from bongo.verify import is_verifiable

csv.field_size_limit(10**9)

EVAL_DIR = DATASET / "eval"
OUT_DIR = DATASET / "eval_dual_script"
REPORT_PATH = DATASET / "metadata" / "stage06_eval_sets_report.json"

# question / answer / extra fields to carry through as metadata
BENCHMARKS = {
    "banglamath": {
        "file": "banglamath/BanglaMath_dataset.csv",
        "format": "csv",
        "question": "Question",
        "answer": "Answer",
        "meta": ["Grade", "Steps", "Digit"],
        "citation": "Prama et al., MathNLP 2025 (arXiv:2510.12836)",
    },
    "bn_mgsm": {
        "file": "bn_mgsm/test-00000-of-00001.parquet",
        "format": "parquet",
        "question": "question",
        "answer": "answer_number",
        "meta": ["equation_solution"],
        "citation": "MGSM Bengali test split",
    },
    "gsm_plus_bn": {
        "file": "gsm_plus_bn/GSM-Plus-BN.csv",
        "format": "csv",
        "question": "Bangla_Question",
        "answer": "Bangla_answer",
        "meta": ["perturbation_type"],
        "citation": "Paul et al., arXiv:2607.13248",
    },
}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_parquet(path):
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    cols = table.column_names
    return [dict(zip(cols, row)) for row in zip(*[table.column(c).to_pylist() for c in cols])]


def build(name, spec):
    path = EVAL_DIR / spec["file"]
    if not path.exists():
        return None, f"missing {spec['file']}"

    rows = read_csv(path) if spec["format"] == "csv" else read_parquet(path)
    records, skipped = [], Counter()

    for i, row in enumerate(rows):
        question = normalize_text(str(row.get(spec["question"], "")))
        answer = str(row.get(spec["answer"], "")).strip()
        if not question:
            skipped["empty_question"] += 1
            continue
        if not answer:
            skipped["empty_answer"] += 1
            continue

        ar, bn = to_ar_digits(question), to_bn_digits(question)
        answer_type = classify(answer)

        records.append({
            "eval_id": f"{name}_{i}",
            "benchmark": name,
            "question_original": question,
            "question_ar_digits": ar,
            "question_bn_digits": bn,
            "variants_differ": ar != bn,
            "gold_answer": answer,
            "gold_answer_ar_digits": to_ar_digits(answer),
            "gold_answer_bn_digits": to_bn_digits(answer),
            "answer_type": answer_type,
            "verifiable": is_verifiable(answer, answer_type),
            "meta": {k: row.get(k) for k in spec["meta"] if row.get(k) not in (None, "")},
        })

    return records, dict(skipped)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {}

    for name, spec in BENCHMARKS.items():
        print("\n" + "=" * 78)
        print(f"{name}  —  {spec['citation']}")
        print("=" * 78)

        records, skipped = build(name, spec)
        if records is None:
            print(f"  SKIPPED: {skipped}")
            report[name] = {"status": "missing", "detail": skipped}
            continue

        n = len(records)
        differ = sum(1 for r in records if r["variants_differ"])
        verifiable = sum(1 for r in records if r["verifiable"])
        gradable = sum(1 for r in records if r["verifiable"] and r["variants_differ"])
        types = Counter(r["answer_type"] for r in records)

        print(f"  problems:                       {n:>7,}")
        print(f"  script variants differ:         {differ:>7,}  ({100*differ/n:.1f}%)")
        print(f"  verifiable answer:              {verifiable:>7,}  ({100*verifiable/n:.1f}%)")
        print(f"  usable for agreement metric:    {gradable:>7,}  ({100*gradable/n:.1f}%)")
        if skipped:
            print(f"  skipped:                        {skipped}")
        print(f"  answer types: {dict(types.most_common(6))}")

        # Which script was the benchmark originally written in? A digit-free
        # question equals both variants, so it must be counted once, separately —
        # otherwise it inflates both buckets and the shares exceed 100%.
        origin = Counter()
        for r in records:
            same_ar = r["question_original"] == r["question_ar_digits"]
            same_bn = r["question_original"] == r["question_bn_digits"]
            if not r["variants_differ"]:
                origin["no_digits"] += 1
            elif same_ar:
                origin["arabic"] += 1
            elif same_bn:
                origin["bengali"] += 1
            else:
                origin["mixed"] += 1
        print("  original script: " + "  ".join(
            f"{k} {100*v/n:.0f}%" for k, v in origin.most_common()
        ))

        out = OUT_DIR / f"{name}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  wrote {out.relative_to(DATASET.parent)}")

        report[name] = {
            "status": "ok",
            "citation": spec["citation"],
            "problems": n,
            "variants_differ": differ,
            "verifiable": verifiable,
            "usable_for_agreement": gradable,
            "answer_types": dict(types),
            "original_script": dict(origin),
            "skipped": skipped,
        }

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {REPORT_PATH.relative_to(DATASET.parent)}")

    total = sum(r.get("usable_for_agreement", 0) for r in report.values())
    print(f"\ntotal problems usable for the agreement metric: {total:,}")


if __name__ == "__main__":
    main()
