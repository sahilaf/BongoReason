"""Stage 1 — build the unified v0.4 record set from the v0.3 master.

Resolves MANIFEST issues B1 (no recoverable final answer), B2 (corrupt
``dart_math_bangla``) and the source-dependent meaning of ``answer``.

Records are routed into two pools rather than one flat set:

  sft  question + Bangla reasoning chain + verified final answer
  rl   question + final answer only (GRPO generates its own chain, so a
       missing or English reference chain is not disqualifying)

Pool membership is derived from the data, not hardcoded per source, so a source
whose chains are fixed later moves pools on its own.

Usage:  python scripts/01_unify_schema.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bongo import DATASET
from bongo.answers import classify, extract_answer
from bongo.normalize import (
    normalize_text,
    script_profile,
    to_ar_digits,
    to_bn_digits,
)
from bongo.verify import is_verifiable

IN_PATH = DATASET / "verified" / "bongo_reason_v0.3.json"
OUT_PATH = DATASET / "interim" / "unified_v0.4.json"
REPORT_PATH = DATASET / "metadata" / "stage01_unify_report.json"

# Sources removed wholesale, with the reason recorded for the data card.
DROP_SOURCES = {
    "dart_math_bangla": (
        "corrupt: translation lost sentence alignment, solution text is shifted "
        "across rows and bleeds between problems (MANIFEST B2). English columns "
        "in the raw CSV are intact if re-translation is ever attempted."
    )
}

# An `answer` field this short with no newline is a bare gold answer, not a
# worked solution. Above it (somadhan) the field holds the whole chain.
BARE_ANSWER_MAX_LEN = 40
BANGLA_THRESHOLD = 0.6


def build_record(r):
    """Return (record, None) or (None, drop_reason)."""
    source = r.get("source", "?")
    if source in DROP_SOURCES:
        return None, "dropped_source"

    question = normalize_text(str(r.get("question_bn", "")))
    if not question:
        return None, "empty_question"

    answer_field = str(r.get("answer", "")).strip()
    solution = normalize_text(str(r.get("solution_bn", "")))

    is_bare = 0 < len(answer_field) <= BARE_ANSWER_MAX_LEN and "\n" not in answer_field

    # somadhan keeps its whole GSM8K-style chain in `answer`; promote it.
    promoted = False
    if not solution and answer_field and not is_bare:
        solution = normalize_text(answer_field)
        promoted = True

    if is_bare:
        raw_answer, method = answer_field, "bare_field"
    else:
        raw_answer, method = extract_answer(solution, answer_field)

    if raw_answer is None:
        return None, "no_recoverable_answer"

    answer_type = classify(raw_answer)
    if answer_type == "missing":
        return None, "no_recoverable_answer"

    flags = []
    q_prof = script_profile(question)
    if q_prof["has_cjk"]:
        flags.append("cjk_in_question")

    solution_lang = None
    if solution:
        s_prof = script_profile(solution)
        if s_prof["has_cjk"]:
            flags.append("cjk_in_solution")
        solution_lang = "bn" if s_prof["bengali_ratio"] > BANGLA_THRESHOLD else "en"

    if any(f.startswith("cjk") for f in flags):
        return None, "cjk_contamination"

    # A chain is only an SFT target if it exists and is actually in Bangla.
    if solution and solution_lang == "bn":
        pool = "sft"
    else:
        pool = "rl"
        if solution_lang == "en":
            flags.append("english_chain_discarded")
        solution = ""

    record = {
        "id": r.get("id"),
        "source": source,
        "pool": pool,
        "question_bn": question,
        "question_ar_digits": to_ar_digits(question),
        "question_bn_digits": to_bn_digits(question),
        "solution_bn": solution or None,
        "solution_ar_digits": to_ar_digits(solution) if solution else None,
        "solution_bn_digits": to_bn_digits(solution) if solution else None,
        "final_answer": raw_answer,
        "answer_type": answer_type,
        "verifiable": is_verifiable(raw_answer, answer_type),
        "answer_method": method,
        "solution_promoted_from_answer": promoted,
        "flags": flags,
    }
    return record, None


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    records = json.loads(IN_PATH.read_text(encoding="utf-8"))
    print(f"loaded {len(records):,} records from {IN_PATH.name}\n")

    kept, drops = [], defaultdict(Counter)
    for r in records:
        rec, reason = build_record(r)
        if rec is None:
            drops[reason][r.get("source", "?")] += 1
        else:
            kept.append(rec)

    print("=" * 78)
    print("DROPPED")
    print("=" * 78)
    total_dropped = 0
    for reason, by_source in sorted(drops.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(by_source.values())
        total_dropped += n
        print(f"  {reason:<26} {n:>6,}   {dict(by_source)}")
    print(f"  {'TOTAL':<26} {total_dropped:>6,}")

    print("\n" + "=" * 78)
    print("KEPT BY POOL AND SOURCE")
    print("=" * 78)
    pools = defaultdict(Counter)
    for r in kept:
        pools[r["pool"]][r["source"]] += 1
    for pool in ("sft", "rl"):
        n = sum(pools[pool].values())
        print(f"\n  {pool.upper()} pool — {n:,} records")
        for source, c in pools[pool].most_common():
            print(f"    {source:<24}{c:>7,}")

    print("\n" + "=" * 78)
    print("ANSWER TYPES  (verifiable types back the Phase 6 RL reward)")
    print("=" * 78)
    types = Counter(r["answer_type"] for r in kept)
    verifiable = sum(1 for r in kept if r["verifiable"])
    for t, c in types.most_common():
        mark = "  <- verifiable" if any(
            r["verifiable"] for r in kept if r["answer_type"] == t
        ) else ""
        print(f"  {t:<14}{c:>7,}{100*c/len(kept):>7.1f}%{mark}")
    print(f"\n  verifiable total: {verifiable:,} / {len(kept):,} ({100*verifiable/len(kept):.1f}%)")

    print("\n  extraction method:")
    for m, c in Counter(r["answer_method"] for r in kept).most_common():
        print(f"    {m:<18}{c:>7,}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(kept, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    report = {
        "stage": "01_unify_schema",
        "input": str(IN_PATH.relative_to(DATASET.parent)),
        "output": str(OUT_PATH.relative_to(DATASET.parent)),
        "input_records": len(records),
        "output_records": len(kept),
        "dropped": {k: dict(v) for k, v in drops.items()},
        "dropped_sources": DROP_SOURCES,
        "by_pool": {p: dict(c) for p, c in pools.items()},
        "answer_types": dict(types),
        "verifiable": verifiable,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nwrote {OUT_PATH.relative_to(DATASET.parent)}  ({len(kept):,} records)")
    print(f"wrote {REPORT_PATH.relative_to(DATASET.parent)}")


if __name__ == "__main__":
    main()
