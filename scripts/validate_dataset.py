"""Validate the built dataset and its splits. Exit code is non-zero on failure,
so this works as a CI gate before any training run.

Replaces the earlier ``audit_dataset.py``, which described the v0.3 master and
does not understand the v0.4 schema. The v0.3 audit it produced is preserved at
``dataset/metadata/audit_v0.3.json``.

Checks:
  * required fields present and correctly typed
  * pool invariants (sft carries a Bangla chain, rl does not)
  * every record flagged ``verifiable`` actually canonicalizes
  * gold chains verify against their own extracted answers
  * splits are disjoint, complete, and reference only existing ids
  * no record survives that appears in an evaluation benchmark

Usage:  python scripts/validate_dataset.py
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bongo import DATASET
from bongo.answers import canonical
from bongo.normalize import is_bangla_dominant, normalize_for_matching
from bongo.verify import verify

csv.field_size_limit(10**9)

DATA_PATH = DATASET / "verified" / "bongo_reason_v0.4.json"
SPLIT_DIR = DATASET / "splits"
EVAL_DIR = DATASET / "eval"

REQUIRED = [
    "id", "source", "pool", "question_bn", "question_ar_digits",
    "question_bn_digits", "final_answer", "answer_type", "verifiable", "flags",
]

failures = []
warnings = []


def fail(msg):
    failures.append(msg)


def warn(msg):
    warnings.append(msg)


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    if not DATA_PATH.exists():
        sys.exit(f"missing {DATA_PATH} — run scripts/run_pipeline.py first")
    records = json.loads(DATA_PATH.read_text(encoding="utf-8"))

    section(f"DATASET — {DATA_PATH.name}")
    print(f"  records: {len(records):,}")
    pools = Counter(r["pool"] for r in records)
    print(f"  by pool: {dict(pools)}")
    print(f"  verifiable: {sum(1 for r in records if r['verifiable']):,}")
    for s, c in Counter(r["source"] for r in records).most_common():
        print(f"    {s:<24}{c:>7,}")

    # ---- schema ----
    section("SCHEMA")
    missing = Counter()
    ids = Counter()
    for r in records:
        for field in REQUIRED:
            if field not in r:
                missing[field] += 1
        ids[str(r["id"])] += 1
    if missing:
        fail(f"records missing required fields: {dict(missing)}")
    dupes = [i for i, c in ids.items() if c > 1]
    if dupes:
        fail(f"{len(dupes)} duplicate ids, e.g. {dupes[:5]}")
    print(f"  required fields present: {'yes' if not missing else 'NO'}")
    print(f"  unique ids: {len(ids):,} / {len(records):,}")

    # ---- pool invariants ----
    section("POOL INVARIANTS")
    sft_no_chain = [r for r in records if r["pool"] == "sft" and not r.get("solution_bn")]
    rl_with_chain = [r for r in records if r["pool"] == "rl" and r.get("solution_bn")]
    if sft_no_chain:
        fail(f"{len(sft_no_chain)} sft records have no solution_bn")
    if rl_with_chain:
        fail(f"{len(rl_with_chain)} rl records unexpectedly carry solution_bn")
    non_bangla = [
        r for r in records
        if r["pool"] == "sft" and not is_bangla_dominant(r["solution_bn"] or "")
    ]
    if non_bangla:
        fail(f"{len(non_bangla)} sft chains are not Bangla-dominant, e.g. {non_bangla[0]['id']}")
    print(f"  sft without chain:      {len(sft_no_chain)}")
    print(f"  rl carrying a chain:    {len(rl_with_chain)}")
    print(f"  sft chains not Bangla:  {len(non_bangla)}")

    # ---- answers ----
    section("ANSWERS")
    bad_canon = [
        r for r in records
        if r["verifiable"] and canonical(r["final_answer"], r["answer_type"]) is None
    ]
    if bad_canon:
        fail(f"{len(bad_canon)} records marked verifiable do not canonicalize")
    print(f"  verifiable but uncanonicalizable: {len(bad_canon)}")

    checkable = [r for r in records if r["pool"] == "sft" and r["verifiable"]]
    self_fail = [
        r for r in checkable
        if not verify(r["solution_bn"], r["final_answer"], r["answer_type"]).correct
    ]
    if self_fail:
        fail(f"{len(self_fail)} gold chains do not verify against their own answer")
    print(f"  gold chains self-verifying: {len(checkable)-len(self_fail):,} / {len(checkable):,}")

    conflicts = [r for r in records if "answer_conflict" in r.get("flags", [])]
    if conflicts:
        warn(f"{len(conflicts)} records kept with unresolved duplicate answer conflicts "
             f"(demoted to unverifiable, safe for SFT)")
    print(f"  unresolved answer conflicts: {len(conflicts)}")

    # ---- splits ----
    section("SPLITS")
    all_ids = {str(r["id"]) for r in records}
    split_ids = {}
    for name in ("train", "val", "test"):
        path = SPLIT_DIR / f"{name}.json"
        if not path.exists():
            fail(f"missing split file {path.name}")
            continue
        split_ids[name] = set(json.loads(path.read_text(encoding="utf-8")))
        print(f"  {name:<6}{len(split_ids[name]):>7,}")

    if len(split_ids) == 3:
        union = set().union(*split_ids.values())
        for a in split_ids:
            for b in split_ids:
                if a < b and split_ids[a] & split_ids[b]:
                    fail(f"splits {a} and {b} overlap on {len(split_ids[a] & split_ids[b])} ids")
        if union != all_ids:
            fail(f"splits cover {len(union):,} ids but dataset has {len(all_ids):,}")
        unknown = union - all_ids
        if unknown:
            fail(f"{len(unknown)} split ids are not in the dataset")
        print(f"  disjoint: {'yes' if not failures else 'see failures'}")
        print(f"  complete coverage: {'yes' if union == all_ids else 'NO'}")

    # ---- contamination ----
    section("CONTAMINATION RE-CHECK")
    eval_keys = set()
    bench_count = 0
    wanted = ("problem", "question", "query", "questions")
    if EVAL_DIR.exists():
        for bench in sorted(p for p in EVAL_DIR.iterdir() if p.is_dir()):
            if bench.name == "gsm_plus_en":
                continue
            for path in bench.glob("*.csv"):
                with path.open(encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    col = next(
                        (c for c in (reader.fieldnames or [])
                         if c.strip().lower() in wanted),
                        None,
                    )
                    if not col:
                        continue
                    bench_count += 1
                    for row in reader:
                        q = str(row.get(col, "")).strip()
                        if q:
                            eval_keys.add(normalize_for_matching(q))
            for path in bench.glob("*.parquet"):
                try:
                    import pyarrow.parquet as pq
                except ImportError:
                    warn("pyarrow not installed; parquet eval sets were not checked")
                    break
                table = pq.read_table(path)
                col = next(
                    (c for c in table.column_names if c.strip().lower() in wanted), None
                )
                if not col:
                    continue
                bench_count += 1
                for q in table.column(col).to_pylist():
                    if str(q).strip():
                        eval_keys.add(normalize_for_matching(str(q)))
    if not eval_keys:
        warn("no evaluation questions available; contamination could not be re-checked")
        print("  SKIPPED — dataset/eval/ is empty")
    else:
        hits = [r for r in records if normalize_for_matching(r["question_bn"]) in eval_keys]
        if hits:
            fail(f"{len(hits)} records still match an evaluation benchmark")
        print(f"  checked against {len(eval_keys):,} eval questions from {bench_count} file(s)")
        print(f"  contaminated records remaining: {len(hits)}")

    gap_path = EVAL_DIR / "manifest.json"
    if gap_path.exists():
        for gap in json.loads(gap_path.read_text(encoding="utf-8")).get(
            "decontamination_gaps", []
        ):
            warn(f"{gap['benchmark']} contamination is still unmeasured — {gap['problem']}")

    # ---- verdict ----
    section("VERDICT")
    for w in warnings:
        print(f"  WARN  {w}")
    for f in failures:
        print(f"  FAIL  {f}")
    if failures:
        print(f"\n  {len(failures)} check(s) failed")
        return 1
    print(f"\n  all checks passed ({len(warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
