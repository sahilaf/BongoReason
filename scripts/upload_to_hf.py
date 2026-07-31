"""Upload the built dataset to a Hugging Face dataset repository.

Defaults to a PRIVATE repo. The derived dataset is built from sources whose
licenses and redistribution terms are not yet established (see "Provenance and
licensing incomplete" in dataset/metadata/MANIFEST.md), so publishing it openly
before that is resolved would distribute material you may not have the right to
redistribute. `--public` exists, but use it deliberately.

Evaluation benchmarks are never uploaded. They belong to their authors, and
re-hosting eval content is how it ends up in someone else's training crawl —
which is the exact failure this project documented in Bn-MGSM.

    python scripts/upload_to_hf.py --repo sahilfarib/bongo-reason
    python scripts/upload_to_hf.py --repo sahilfarib/bongo-reason --public
    python scripts/upload_to_hf.py --repo sahilfarib/bongo-reason --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bongo import DATASET, ROOT

# (local path, path inside the HF repo). Directories upload recursively.
UPLOADS = [
    (DATASET / "verified" / "bongo_reason_v0.4.json", "data/bongo_reason_v0.4.json"),
    (DATASET / "script_pairs" / "script_pairs_v0.4.json", "data/script_pairs_v0.4.json"),
    (DATASET / "corrections" / "error_correction_pairs_v0.1.json",
     "data/error_correction_pairs_v0.1.json"),
    (DATASET / "splits", "splits"),
    (DATASET / "metadata", "metadata"),
]

# Never upload: raw sources (2 GB, unclear licensing) and any evaluation
# benchmark or derivative of one.
NEVER = ["dataset/raw", "dataset/eval", "dataset/eval_dual_script", "results"]


def build_card(repo, public):
    stage3 = json.loads(
        (DATASET / "metadata" / "stage03_decontamination_report.json").read_text(encoding="utf-8")
    )
    n = stage3["output_records"]
    visibility = "public" if public else "private"

    return f"""---
license: other
license_name: mixed-see-manifest
language:
- bn
task_categories:
- question-answering
- text-generation
tags:
- mathematical-reasoning
- bangla
- bengali
- low-resource
- dual-script
size_categories:
- 10K<n<100K
---

# BongoReason v0.4 — Bangla Mathematical Reasoning

{n:,} Bangla math reasoning records with verified final answers, frozen splits,
and dual-script (Bengali/Arabic numeral) pairs.

Code and full documentation: see the project repository.
Data card: `metadata/MANIFEST.md` in this repo.

> **Status: {visibility}.** Licensing of the underlying sources is not fully
> established — see the Licensing section below. Do not redistribute without
> resolving it.

## Contents

| File | Description |
|---|---|
| `data/bongo_reason_v0.4.json` | the dataset ({n:,} records) |
| `data/script_pairs_v0.4.json` | 22,469 dual-script question pairs |
| `data/error_correction_pairs_v0.1.json` | 100 rule-based error/correction pairs |
| `splits/` | frozen train/val/test id manifests |
| `metadata/` | data card and per-stage pipeline reports |

## Pools

Records are routed by what they can support, derived from the data rather than
hardcoded per source:

- **`sft`** (22,837) — question + Bangla reasoning chain + final answer
- **`rl`** (1,062) — question + final answer only

17,570 records carry a **verifiable** answer, meaning one a deterministic
verifier can check exactly (`numeric`, `fraction`, `percent`, `mcq_option`,
`bool_bn`). `expression` and `symbolic` answers are SFT-only — exact LaTeX
equivalence checking is unreliable, and a flaky reward is worse than a narrow one.

Note that GRPO prompt pools are not limited to the `rl` pool: every verifiable
record qualifies, since GRPO needs only a question and a checkable answer.

## Provenance

Built from 874,467 raw records across 8 sources. 6,249 were removed:

| Removed | Reason |
|---:|---|
| 1,000 | `dart_math_bangla` — translation lost sentence alignment, text shifted across rows |
| 768 | no recoverable final answer after all extraction strategies |
| 202 | Chinese text leaked from NuminaMath's Chinese-origin problems |
| 594 | exact and near-duplicate questions |
| 730 | **perturbed forms of Bn-MGSM test problems** |
| 2,946 | lineage unverifiable — parent benchmark unavailable |

### The contamination finding

`distractmath_mgsm` is built by injecting distractors into MGSM problems. All 738
of its rows derive from exactly the 250 **Bn-MGSM test** questions. Because the
stored question is the perturbed form, only **8 of 738 matched textually** —
surface decontamination misses 99% of this.

If you train on any perturbation-augmented dataset and evaluate on its source
benchmark, string or MinHash decontamination will not protect you. Check lineage.

## Dual-script pairs

Every question is stored in both numeral scripts. Pairs whose variants are
identical (no digits in the question) are excluded — they cannot disagree, so
including them would inflate any consistency metric.

Source corpora differ sharply in native script, which is what motivates the
work: `bengali_math_cot` and `numina_bn` are ~83% Arabic-digit while `somadhan`
is 98% Bengali-digit.

## Licensing

**This is unresolved and you should read this before using the data.**

The pipeline code is MIT. This dataset is *not* covered by that. It derives from
third-party sources whose licenses and redistribution terms have not all been
established. Known: `kawchar85/Bangla-Math` (MIT), MGSM (CC-BY-SA-4.0). Unknown:
`bengali_math_cot`, `somadhan`, `numina_bn` translations.

`somadhan` alone is 3,994 records (17%) and its source paper has not been located.

Evaluation benchmarks are **not** included here. Fetch them from their original
sources: BanglaMATH (arXiv:2510.12836), GSM-Plus-BN
([Mendeley](https://data.mendeley.com/datasets/74dscnmrhv/3)), MGSM, BenNumEval.

## Reproducing

```bash
python scripts/run_pipeline.py
```

All hashing is `blake2b` rather than Python's salted builtin `hash()`, so builds
are byte-identical across runs and machines.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. username/bongo-reason")
    ap.add_argument("--public", action="store_true",
                    help="create a PUBLIC repo (default private; see module docstring)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    from huggingface_hub import HfApi

    api = HfApi()
    try:
        who = api.whoami()["name"]
    except Exception as e:
        sys.exit(f"not authenticated: {e}\nRun:  huggingface-cli login")
    print(f"authenticated as {who}")
    print(f"target: {args.repo}  ({'PUBLIC' if args.public else 'private'})\n")

    plan, total, missing = [], 0, []
    for local, remote in UPLOADS:
        if not local.exists():
            missing.append(str(local.relative_to(ROOT)))
            continue
        if local.is_dir():
            for f in sorted(local.rglob("*")):
                if f.is_file():
                    plan.append((f, f"{remote}/{f.relative_to(local).as_posix()}", f.stat().st_size))
                    total += f.stat().st_size
        else:
            plan.append((local, remote, local.stat().st_size))
            total += local.stat().st_size

    print(f"{'local':<52}{'-> repo path':<44}{'MB':>8}")
    for local, remote, size in plan:
        print(f"{str(local.relative_to(ROOT)):<52}{remote:<44}{size/1024**2:>8.1f}")
    print(f"\n{len(plan)} files, {total/1024**2:.1f} MB total")
    print(f"never uploaded: {NEVER}")

    if missing:
        print(f"\nMISSING (run scripts/run_pipeline.py first): {missing}")
        return 1

    if args.dry_run:
        print("\ndry run — nothing uploaded")
        return 0

    if args.public:
        print("\n" + "!" * 78)
        print("Creating a PUBLIC dataset repo. Source licensing is unresolved")
        print("(see dataset/metadata/MANIFEST.md). Publishing distributes this material.")
        print("!" * 78)

    api.create_repo(args.repo, repo_type="dataset",
                    private=not args.public, exist_ok=True)
    print(f"\nrepo ready: https://huggingface.co/datasets/{args.repo}")

    card = ROOT / "HF_DATASET_CARD.md"
    card.write_text(build_card(args.repo, args.public), encoding="utf-8")
    api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="dataset")
    card.unlink()
    print("uploaded dataset card")

    for i, (local, remote, size) in enumerate(plan, 1):
        print(f"  [{i}/{len(plan)}] {remote}  ({size/1024**2:.1f} MB)", flush=True)
        api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                        repo_id=args.repo, repo_type="dataset")

    print(f"\ndone: https://huggingface.co/datasets/{args.repo}")
    if not args.public:
        print("Repo is PRIVATE. Make it public from the repo settings once you have")
        print("resolved source licensing (MANIFEST 'Provenance and licensing incomplete').")
    return 0


if __name__ == "__main__":
    sys.exit(main())
