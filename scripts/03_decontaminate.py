"""Stage 3 — remove training records that appear in an evaluation benchmark
(MANIFEST B5).

Matching is exact on the numeral-script-insensitive form, plus MinHash near
matching at a deliberately loose threshold: a benchmark problem reworded
slightly is still contamination, so this stage errs toward removing.

The stage refuses to run against an empty ``dataset/eval/``.  A decontamination
step that silently passes because it had nothing to compare against is worse
than no step at all — it produces a clean-looking report and a contaminated
model.

Usage:  python scripts/03_decontaminate.py [--threshold 0.8]
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bongo import DATASET
from bongo.dedup import cross_candidates, jaccard, signatures
from bongo.normalize import normalize_for_matching, shingles

csv.field_size_limit(10**9)

IN_PATH = DATASET / "interim" / "deduped_v0.4.json"
OUT_PATH = DATASET / "verified" / "bongo_reason_v0.4.json"
REPORT_PATH = DATASET / "metadata" / "stage03_decontamination_report.json"
EVAL_DIR = DATASET / "eval"

QUESTION_COLUMN_HINTS = (
    "problem", "question", "questions", "query", "input", "text",
    "original_question", "modified_question",
)

# Benchmarks fetched for reference but not usable for string matching against
# Bangla training text.
SKIP_FOR_MATCHING = {"gsm_plus_en"}

# Sources built by *perturbing* a benchmark. Their stored question is the
# perturbed form, which does not string-match the benchmark even though the
# problem is the same one — 738/738 distractmath_mgsm rows derive from the 250
# Bn-MGSM test questions, but only 8 match textually. Checking lineage instead
# of surface text is the only way to catch this.
#
# raw_file / parent_column / derived_column let stage 3 rebuild the mapping from
# the perturbed question back to the benchmark question it came from.
DERIVED_SOURCES = {
    "distractmath_mgsm": {
        "raw_file": "distractmath_mgsm_raw.csv",
        "parent_column": "original_question",
        "derived_column": "modified_question",
        "parent_benchmark": "bn_mgsm",
    },
    "distractmath_msvamp": {
        "raw_file": "distractmath_msvamp_raw.csv",
        "parent_column": "original_question",
        "derived_column": "modified_question",
        "parent_benchmark": "bn_msvamp",
    },
}


def sniff_question_column(fieldnames):
    for hint in QUESTION_COLUMN_HINTS:
        for name in fieldnames:
            if name.strip().lower() == hint:
                return name
    for hint in QUESTION_COLUMN_HINTS:
        for name in fieldnames:
            if hint in name.strip().lower():
                return name
    return None


def all_question_columns(fieldnames):
    """Every column that looks like a question.

    GSM-Plus-BN ships both the perturbed question and the seed question it was
    derived from. Both matter for contamination: training data translated from
    GSM8K will match the *seed*, not the perturbation. Collecting only one column
    would silently miss that.
    """
    cols = [
        name for name in (fieldnames or [])
        if name and any(h in name.strip().lower() for h in ("question", "problem", "query"))
    ]
    return cols or ([sniff_question_column(fieldnames)] if sniff_question_column(fieldnames) else [])


def read_parquet_questions(path):
    """Extract the question column from a parquet file."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print(f"  ! {path.name}: pyarrow not installed, skipping (pip install pyarrow)")
        return []
    table = pq.read_table(path)
    col = sniff_question_column(table.column_names)
    if not col:
        print(f"  ! {path.name}: no question column in {table.column_names}")
        return []
    return [str(v) for v in table.column(col).to_pylist() if str(v).strip()]


def load_eval_questions():
    """Return {benchmark_name: [question, ...]} for every set under dataset/eval/."""
    out = {}
    if not EVAL_DIR.exists():
        return out
    for bench_dir in sorted(p for p in EVAL_DIR.iterdir() if p.is_dir()):
        if bench_dir.name in SKIP_FOR_MATCHING:
            continue
        questions = []
        for path in sorted(bench_dir.iterdir()):
            if path.suffix.lower() == ".parquet":
                questions += read_parquet_questions(path)
            elif path.suffix.lower() in (".tsv", ".csv"):
                delim = "\t" if path.suffix.lower() == ".tsv" else ","
                with path.open(encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f, delimiter=delim)
                    cols = all_question_columns(reader.fieldnames or [])
                    if not cols:
                        print(f"  ! {path.name}: no question column in {reader.fieldnames}")
                        continue
                    if len(cols) > 1:
                        print(f"    {path.name}: using {len(cols)} question columns {cols}")
                    for row in reader:
                        for col in cols:
                            if str(row.get(col, "")).strip():
                                questions.append(str(row[col]))
            elif path.suffix.lower() in (".jsonl", ".json"):
                text = path.read_text(encoding="utf-8")
                rows = (
                    [json.loads(line) for line in text.splitlines() if line.strip()]
                    if path.suffix.lower() == ".jsonl"
                    else json.loads(text)
                )
                if isinstance(rows, dict):
                    rows = rows.get("data", [])
                if rows and isinstance(rows[0], dict):
                    col = sniff_question_column(list(rows[0].keys()))
                    if col:
                        questions += [
                            str(r[col]) for r in rows if str(r.get(col, "")).strip()
                        ]
        if questions:
            out[bench_dir.name] = questions
    return out


def load_lineage():
    """Map normalized perturbed question -> normalized parent question, per source."""
    lineage = {}
    for source, spec in DERIVED_SOURCES.items():
        path = DATASET / "raw" / spec["raw_file"]
        if not path.exists():
            continue
        mapping = {}
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                derived = str(row.get(spec["derived_column"], "")).strip()
                parent = str(row.get(spec["parent_column"], "")).strip()
                if derived and parent:
                    mapping[normalize_for_matching(derived)] = normalize_for_matching(parent)
        lineage[source] = {"map": mapping, "benchmark": spec["parent_benchmark"]}
    return lineage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="Jaccard threshold; looser than dedup on purpose")
    ap.add_argument("--keep-unverifiable-lineage", action="store_true",
                    help="keep derived records whose parent benchmark is missing "
                         "(default: drop them, since lineage cannot be cleared)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    records = json.loads(IN_PATH.read_text(encoding="utf-8"))
    print(f"loaded {len(records):,} training records\n")

    print("=" * 78)
    print("EVALUATION SETS")
    print("=" * 78)
    eval_sets = load_eval_questions()
    if not eval_sets:
        sys.exit(
            "\nFATAL: no evaluation questions found under dataset/eval/.\n"
            "Run  python scripts/00_fetch_eval.py  first.\n"
            "Refusing to emit a 'decontaminated' dataset that was never checked."
        )
    for name, qs in eval_sets.items():
        print(f"  {name:<24}{len(qs):>7,} questions")

    # ---- exact ----
    eval_keys = {}
    for name, qs in eval_sets.items():
        for q in qs:
            eval_keys.setdefault(normalize_for_matching(q), name)

    contaminated = {}
    for idx, r in enumerate(records):
        bench = eval_keys.get(normalize_for_matching(r["question_bn"]))
        if bench:
            contaminated[idx] = (bench, "exact")
    print(f"\n  exact matches: {len(contaminated):,}")

    # ---- near ----
    eval_texts, eval_owner = [], []
    for name, qs in eval_sets.items():
        eval_texts += qs
        eval_owner += [name] * len(qs)
    eval_shingles = [shingles(q) for q in eval_texts]
    train_shingles = [shingles(r["question_bn"]) for r in records]

    print(f"  building MinHash over {len(eval_texts):,} eval + {len(records):,} train ...")
    eval_sigs = signatures(eval_shingles)
    train_sigs = signatures(train_shingles)

    near = 0
    for i, j in cross_candidates(train_sigs, eval_sigs):
        if i in contaminated:
            continue
        if jaccard(train_shingles[i], eval_shingles[j]) >= args.threshold:
            contaminated[i] = (eval_owner[j], "near")
            near += 1
    print(f"  near matches (J>={args.threshold}): {near:,}")

    # ---- lineage ----
    lineage = load_lineage()
    lineage_hits = 0
    unverifiable = Counter()
    for idx, r in enumerate(records):
        spec = lineage.get(r["source"])
        if not spec or idx in contaminated:
            continue
        parent = spec["map"].get(normalize_for_matching(r["question_bn"]))
        bench = spec["benchmark"]
        if parent is None:
            continue
        if parent in eval_keys:
            contaminated[idx] = (bench, "lineage")
            lineage_hits += 1
        elif bench not in eval_sets:
            # The parent benchmark was never downloaded, so this record's lineage
            # cannot be cleared. Dropping is the safe default.
            unverifiable[r["source"]] += 1
            if not args.keep_unverifiable_lineage:
                contaminated[idx] = (f"{bench}(unavailable)", "lineage_unverifiable")
    print(f"  lineage matches (perturbed form of a benchmark problem): {lineage_hits:,}")
    if unverifiable:
        action = "kept" if args.keep_unverifiable_lineage else "dropped"
        print(f"  lineage unverifiable, parent benchmark missing ({action}): {dict(unverifiable)}")

    # ---- result ----
    print("\n" + "=" * 78)
    print("CONTAMINATION")
    print("=" * 78)
    by_source = Counter(records[i]["source"] for i in contaminated)
    by_bench = Counter(b for b, _ in contaminated.values())
    by_kind = Counter(k for _, k in contaminated.values())
    print(f"  contaminated records: {len(contaminated):,} / {len(records):,} "
          f"({100*len(contaminated)/len(records):.1f}%)")
    print(f"  by benchmark: {dict(by_bench)}")
    print(f"  by match kind: {dict(by_kind)}")
    print("  by training source:")
    source_totals = Counter(r["source"] for r in records)
    for s, c in by_source.most_common():
        print(f"    {s:<24}{c:>7,} of {source_totals[s]:,} ({100*c/source_totals[s]:.0f}% of source)")

    clean = [r for i, r in enumerate(records) if i not in contaminated]
    fully_removed = [s for s in source_totals if by_source[s] == source_totals[s]]
    if fully_removed:
        print(f"\n  sources removed entirely: {fully_removed}")

    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    pools = Counter(r["pool"] for r in clean)
    print(f"  {len(records):,} -> {len(clean):,}")
    print(f"  by pool:    {dict(pools)}")
    print(f"  verifiable: {sum(1 for r in clean if r['verifiable']):,}")
    for s, c in Counter(r["source"] for r in clean).most_common():
        print(f"    {s:<24}{c:>7,}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(clean, ensure_ascii=False, indent=1), encoding="utf-8")

    gaps = []
    manifest_path = EVAL_DIR / "manifest.json"
    if manifest_path.exists():
        gaps = json.loads(manifest_path.read_text(encoding="utf-8")).get(
            "decontamination_gaps", []
        )

    REPORT_PATH.write_text(
        json.dumps(
            {
                "stage": "03_decontaminate",
                "threshold": args.threshold,
                "eval_sets": {k: len(v) for k, v in eval_sets.items()},
                "input_records": len(records),
                "contaminated": len(contaminated),
                "by_benchmark": dict(by_bench),
                "by_match_kind": dict(by_kind),
                "by_source": dict(by_source),
                "lineage_matches": lineage_hits,
                "lineage_unverifiable": dict(unverifiable),
                "sources_removed_entirely": fully_removed,
                "output_records": len(clean),
                "by_pool": dict(pools),
                "verifiable": sum(1 for r in clean if r["verifiable"]),
                "unresolved_gaps": gaps,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {OUT_PATH.relative_to(DATASET.parent)}")
    print(f"wrote {REPORT_PATH.relative_to(DATASET.parent)}")

    if gaps:
        print("\n" + "!" * 78)
        for g in gaps:
            print(f"STILL UNCHECKED: {g['benchmark']} — {g['problem']}")
        print("!" * 78)


if __name__ == "__main__":
    main()
