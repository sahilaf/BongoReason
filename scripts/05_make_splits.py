"""Stage 5 — assign frozen train/val/test splits (MANIFEST B6).

Splits are deliberately generated *last*, on data that is already unified,
deduplicated and decontaminated, because a split assigned before those steps has
to be thrown away when they run.

Assignment is stratified over (pool, source, answer_type, verifiable) so every
experiment sees the same mix, and it is keyed on a hash of the record id rather
than a shuffle — adding records later does not reshuffle the existing ones, so
a split stays comparable across dataset versions.

The internal test split is small on purpose: the headline numbers come from
BanglaMath and GSM-Plus-BN, and this split only exists to catch regressions
without touching those.

Usage:  python scripts/05_make_splits.py [--val 0.05] [--test 0.05]
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bongo import DATASET

IN_PATH = DATASET / "verified" / "bongo_reason_v0.4.json"
OUT_DIR = DATASET / "splits"
REPORT_PATH = DATASET / "metadata" / "stage05_splits_report.json"

SALT = "bongoreason-v0.4"


def bucket(record_id, salt=SALT):
    """Stable [0, 1) position for a record id."""
    digest = hashlib.blake2b(f"{salt}:{record_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=float, default=0.05)
    ap.add_argument("--test", type=float, default=0.05)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    records = json.loads(IN_PATH.read_text(encoding="utf-8"))
    print(f"loaded {len(records):,} records\n")

    # Stratify: assign within each stratum so small strata are not swallowed
    # whole by one split.
    strata = defaultdict(list)
    for r in records:
        strata[(r["pool"], r["source"], r["answer_type"], bool(r["verifiable"]))].append(r)

    assignment = {}
    for key, group in strata.items():
        group = sorted(group, key=lambda r: bucket(r["id"]))
        n = len(group)
        n_val = round(n * args.val)
        n_test = round(n * args.test)
        # A stratum too small to sample from stays entirely in train rather than
        # donating its only record to test.
        if n < 10:
            n_val = n_test = 0
        for i, r in enumerate(group):
            if i < n_test:
                assignment[r["id"]] = "test"
            elif i < n_test + n_val:
                assignment[r["id"]] = "val"
            else:
                assignment[r["id"]] = "train"

    splits = defaultdict(list)
    for r in records:
        splits[assignment[r["id"]]].append(r)

    print("=" * 78)
    print("SPLITS")
    print("=" * 78)
    for name in ("train", "val", "test"):
        rs = splits[name]
        pools = Counter(r["pool"] for r in rs)
        print(f"  {name:<6}{len(rs):>7,} ({100*len(rs)/len(records):>4.1f}%)   "
              f"sft={pools['sft']:,} rl={pools['rl']:,}   "
              f"verifiable={sum(1 for r in rs if r['verifiable']):,}")

    print("\n  source distribution (should match across splits):")
    sources = sorted({r["source"] for r in records})
    print(f"    {'source':<24}" + "".join(f"{n:>9}" for n in ("train", "val", "test")))
    for s in sources:
        row = f"    {s:<24}"
        for name in ("train", "val", "test"):
            rs = splits[name]
            c = sum(1 for r in rs if r["source"] == s)
            row += f"{100*c/len(rs):>8.1f}%"
        print(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for name in ("train", "val", "test"):
        ids = sorted(str(r["id"]) for r in splits[name])
        path = OUT_DIR / f"{name}.json"
        path.write_text(json.dumps(ids, ensure_ascii=False, indent=0), encoding="utf-8")
        checksums[name] = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()[:16]
        print(f"\n  wrote {path.relative_to(DATASET.parent)}  "
              f"({len(ids):,} ids, sha256:{checksums[name]})")

    # Pool-specific id lists, since SFT and RL train on different subsets.
    for pool in ("sft", "rl"):
        for name in ("train", "val", "test"):
            ids = sorted(str(r["id"]) for r in splits[name] if r["pool"] == pool)
            (OUT_DIR / f"{pool}_{name}.json").write_text(
                json.dumps(ids, ensure_ascii=False, indent=0), encoding="utf-8"
            )
    print(f"  wrote per-pool id lists (sft_*/rl_*) to {OUT_DIR.relative_to(DATASET.parent)}")

    REPORT_PATH.write_text(
        json.dumps(
            {
                "stage": "05_make_splits",
                "salt": SALT,
                "val_fraction": args.val,
                "test_fraction": args.test,
                "total": len(records),
                "counts": {k: len(v) for k, v in splits.items()},
                "by_pool": {
                    k: dict(Counter(r["pool"] for r in v)) for k, v in splits.items()
                },
                "verifiable": {
                    k: sum(1 for r in v if r["verifiable"]) for k, v in splits.items()
                },
                "checksums": checksums,
                "strata": len(strata),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  wrote {REPORT_PATH.relative_to(DATASET.parent)}")
    print("\n  These id lists are frozen. Do not regenerate them for a new "
          "experiment;\n  regenerate only when the underlying dataset version changes.")


if __name__ == "__main__":
    main()
