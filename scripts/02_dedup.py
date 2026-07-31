"""Stage 2 — exact and near-duplicate removal (MANIFEST B4).

Exact matching runs on ``normalize_for_matching``, which collapses numeral
script, so a problem written in Bengali digits collides with the same problem in
Arabic digits.

Near-duplicate detection is MinHash + LSH banding over character 5-shingles
(see ``bongo.dedup``).  Candidate pairs from banding are always confirmed with
an exact Jaccard before merging.

Usage:  python scripts/02_dedup.py [--threshold 0.9]
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bongo import DATASET
from bongo.answers import canonical
from bongo.dedup import BANDS, NUM_PERMS, ROWS, Union, candidate_pairs, jaccard, signatures
from bongo.normalize import normalize_for_matching, shingles

IN_PATH = DATASET / "interim" / "unified_v0.4.json"
OUT_PATH = DATASET / "interim" / "deduped_v0.4.json"
REPORT_PATH = DATASET / "metadata" / "stage02_dedup_report.json"


def _preference(r):
    """Most usable signal first: SFT chain > RL prompt, verifiable > not, longer
    chain > shorter.  Id breaks ties so the choice is deterministic."""
    return (
        r["pool"] != "sft",
        not r["verifiable"],
        -len(r["solution_bn"] or ""),
        str(r["id"]),
    )


def keeper(records, conflicts=None):
    """Pick the survivor of a duplicate cluster.

    Duplicates sometimes disagree on the gold answer.  Silently keeping an
    arbitrary one would seed the RL reward with wrong targets, so the cluster
    votes: the majority answer wins, and a cluster with no majority keeps its
    best record but is demoted to unverifiable and flagged for review.
    """
    if len(records) == 1:
        return records[0]

    votes = Counter()
    for r in records:
        c = canonical(r["final_answer"], r["answer_type"])
        if c is not None:
            votes[(r["answer_type"], repr(c))] += 1

    if len(votes) > 1:
        (top, top_n), (_, runner_n) = (votes.most_common(2) + [((None, None), 0)])[:2]
        if top_n > runner_n:
            agreeing = [
                r
                for r in records
                if (r["answer_type"], repr(canonical(r["final_answer"], r["answer_type"])))
                == top
            ]
            if conflicts is not None:
                conflicts["resolved_by_majority"] += 1
            return min(agreeing, key=_preference)

        chosen = dict(min(records, key=_preference))
        chosen["verifiable"] = False
        chosen["flags"] = list(chosen["flags"]) + ["answer_conflict"]
        if conflicts is not None:
            conflicts["unresolved_demoted"] += 1
        return chosen

    return min(records, key=_preference)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.9)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    records = json.loads(IN_PATH.read_text(encoding="utf-8"))
    print(f"loaded {len(records):,} records\n")

    conflicts = Counter()

    # ---- exact ----
    groups = defaultdict(list)
    for r in records:
        groups[normalize_for_matching(r["question_bn"])].append(r)
    exact_survivors = [keeper(g, conflicts) for g in groups.values()]
    exact_removed = len(records) - len(exact_survivors)
    print("=" * 78)
    print("EXACT DUPLICATES (numeral-script insensitive)")
    print("=" * 78)
    print(f"  groups: {len(groups):,}   removed: {exact_removed:,}")
    cross_source = sum(1 for g in groups.values() if len({r['source'] for r in g}) > 1)
    print(f"  duplicate groups spanning more than one source: {cross_source:,}")

    # ---- near ----
    print("\n" + "=" * 78)
    print(f"NEAR DUPLICATES (MinHash {NUM_PERMS} perms, {BANDS}x{ROWS} bands, J>={args.threshold})")
    print("=" * 78)
    docs = [shingles(r["question_bn"]) for r in exact_survivors]
    sigs = signatures(docs)
    union = Union(len(exact_survivors))

    checked = merged = 0
    for i, j in candidate_pairs(sigs):
        checked += 1
        if jaccard(docs[i], docs[j]) >= args.threshold:
            union.union(i, j)
            merged += 1
    print(f"  candidate pairs examined: {checked:,}   pairs merged: {merged:,}")

    clusters = defaultdict(list)
    for idx, r in enumerate(exact_survivors):
        clusters[union.find(idx)].append(r)
    survivors = [keeper(c, conflicts) for c in clusters.values()]
    near_removed = len(exact_survivors) - len(survivors)
    multi = sum(1 for c in clusters.values() if len(c) > 1)
    print(f"  near-duplicate clusters: {multi:,}   removed: {near_removed:,}")

    print("\n" + "=" * 78)
    print("GOLD-ANSWER CONFLICTS WITHIN DUPLICATE CLUSTERS")
    print("=" * 78)
    print(f"  resolved by majority vote:        {conflicts['resolved_by_majority']:,}")
    print(f"  no majority, demoted to unverified: {conflicts['unresolved_demoted']:,}")

    # ---- report ----
    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"  {len(records):,} -> {len(survivors):,}  "
          f"(-{len(records)-len(survivors):,}, "
          f"{100*(len(records)-len(survivors))/len(records):.1f}%)")
    by_pool = Counter(r["pool"] for r in survivors)
    by_source = Counter(r["source"] for r in survivors)
    print(f"  by pool:   {dict(by_pool)}")
    print(f"  verifiable: {sum(1 for r in survivors if r['verifiable']):,}")
    print("  by source:")
    for s, c in by_source.most_common():
        print(f"    {s:<24}{c:>7,}")

    survivors.sort(key=lambda r: (r["source"], str(r["id"])))
    OUT_PATH.write_text(json.dumps(survivors, ensure_ascii=False, indent=1), encoding="utf-8")
    REPORT_PATH.write_text(
        json.dumps(
            {
                "stage": "02_dedup",
                "threshold": args.threshold,
                "permutations": NUM_PERMS,
                "bands": BANDS,
                "rows_per_band": ROWS,
                "input_records": len(records),
                "exact_removed": exact_removed,
                "near_removed": near_removed,
                "output_records": len(survivors),
                "by_pool": dict(by_pool),
                "by_source": dict(by_source),
                "verifiable": sum(1 for r in survivors if r["verifiable"]),
                "answer_conflicts": dict(conflicts),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {OUT_PATH.relative_to(DATASET.parent)}")
    print(f"wrote {REPORT_PATH.relative_to(DATASET.parent)}")


if __name__ == "__main__":
    main()
