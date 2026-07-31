"""Stage 0 — fetch evaluation benchmarks into ``dataset/eval/``.

Every benchmark here must be present before stage 3, because decontamination is
only as good as the eval sets it can see.

## Benchmark identity — read this before editing

``dataset/raw/bangla_math_bdmo_raw.csv`` is byte-identical to the HuggingFace
dataset ``kawchar85/Bangla-Math``. That dataset is **BdMO olympiad problems with
synthetically generated CoT/PoT annotations, described by its own card as a
training dataset with no associated paper**. It is *not* BanglaMATH.

BanglaMATH is Prama, Danforth & Dodds (MathNLP 2025, arXiv 2510.12836): 1,700
grade 6–8 problems from Bangla school workbooks. Different authors, different
size, different source, different purpose. The names collide; the artifacts do
not. An earlier version of this pipeline conflated them and removed 997
legitimate training records by matching them against a copy of themselves.

Usage:  python scripts/00_fetch_eval.py
"""

import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bongo import DATASET

EVAL_DIR = DATASET / "eval"

BANGLAMATH_RAW = (
    "https://raw.githubusercontent.com/TabiaTanzin/"
    "BanglaMATH-A-Bangla-benchmark-dataset-for-testing-LLM-mathematical-"
    "reasoning-at-grades-6-7-and-8/main/"
)

# Fetched from the Hub with huggingface_hub.
HF_BENCHMARKS = [
    {
        "name": "bn_mgsm",
        "repo": "juletxara/mgsm",
        "files": ["bn/test-00000-of-00001.parquet", "mgsm_bn.tsv"],
        "any_of": True,  # layouts differ between revisions; take whichever works
        "license": "cc-by-sa-4.0",
        "role": "primary",
        "note": "MGSM Bengali test split, 250 problems. GanitLLM reports on this.",
    },
    {
        "name": "bennumeval",
        "repo": "ka05ar/BenNumEval",
        "files": [
            "Task1(CA).csv", "Task2(DS).csv", "Task3(CQ).csv",
            "Task4(FiB).csv", "Task5(QNLI).csv", "Task6(AWP).csv",
        ],
        "license": "mit",
        "role": "secondary",
        "note": "Gated repo — request access at the dataset page.",
    },
    {
        "name": "gsm_plus_en",
        "repo": "qintongli/GSM-Plus",
        "files": ["data/test-00000-of-00001.jsonl"],
        "license": "cc-by-sa-4.0",
        "role": "reference_only",
        "note": "English source of GSM-Plus-BN. Cannot string-match Bangla text.",
    },
]

# Fetched over plain HTTP.
URL_BENCHMARKS = [
    {
        "name": "banglamath",
        "urls": {
            "BanglaMath_dataset.csv": BANGLAMATH_RAW + "BanglaMath%20-%20Bangla_Math_dataset.csv",
            "BanglaMath_distractors.csv": BANGLAMATH_RAW + "BanglaMath%20-%20Distractors_dataset.csv",
        },
        "license": "cc-by-4.0",
        "role": "primary",
        "note": "Prama et al., MathNLP 2025. 1,700 grade 6-8 problems.",
    },
]

# Benchmarks that cannot be fetched automatically. Listed so the gap is visible
# in the manifest rather than silently absent.
MANUAL_BENCHMARKS = [
    {
        "name": "gsm_plus_bn",
        "role": "primary",
        "source": "https://data.mendeley.com/datasets/74dscnmrhv/3",
        "citation": "Paul et al., arXiv:2607.13248 (July 2026)",
        "size": "10,544 instances from 1,318 seed questions",
        "instructions": (
            "Mendeley Data requires an interactive download. Fetch v3 from the URL "
            "above and unpack it into dataset/eval/gsm_plus_bn/. Stage 3 will pick "
            "up any CSV/JSON there automatically."
        ),
    },
]


def fetch_hf(bench, out_dir):
    from huggingface_hub import hf_hub_download

    got = []
    for fname in bench["files"]:
        try:
            cached = hf_hub_download(bench["repo"], fname, repo_type="dataset")
        except Exception as e:
            if not bench.get("any_of"):
                print(f"  FAILED {fname}: {type(e).__name__}")
            continue
        dest = out_dir / Path(fname).name
        shutil.copyfile(cached, dest)
        print(f"  {dest.name:<38}{dest.stat().st_size/1024:>10,.1f} KB")
        got.append(dest.name)
        if bench.get("any_of"):
            break
    if bench.get("any_of") and not got:
        print(f"  FAILED — none of {bench['files']} available")
    return got


def fetch_urls(bench, out_dir):
    import urllib.request

    got = []
    for name, url in bench["urls"].items():
        dest = out_dir / name
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
        except Exception as e:
            print(f"  FAILED {name}: {type(e).__name__}: {e}")
            continue
        dest.write_bytes(data)
        print(f"  {name:<38}{len(data)/1024:>10,.1f} KB")
        got.append(name)
    return got


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for bench in URL_BENCHMARKS:
        out_dir = EVAL_DIR / bench["name"]
        out_dir.mkdir(exist_ok=True)
        print(f"\n=== {bench['name']}  ({bench['role']}) ===")
        print(f"    {bench['note']}")
        manifest.append({**bench, "downloaded": fetch_urls(bench, out_dir)})

    for bench in HF_BENCHMARKS:
        out_dir = EVAL_DIR / bench["name"]
        out_dir.mkdir(exist_ok=True)
        print(f"\n=== {bench['name']}  ({bench['repo']}, {bench['role']}) ===")
        print(f"    {bench['note']}")
        manifest.append({**bench, "downloaded": fetch_hf(bench, out_dir)})

    for bench in MANUAL_BENCHMARKS:
        out_dir = EVAL_DIR / bench["name"]
        out_dir.mkdir(exist_ok=True)
        present = [p.name for p in out_dir.iterdir() if p.is_file()]
        status = f"present ({len(present)} file(s))" if present else "NOT PRESENT"
        print(f"\n=== {bench['name']}  (manual, {bench['role']}) — {status} ===")
        if not present:
            print(f"    {bench['instructions']}")
        manifest.append({**bench, "downloaded": present})

    missing = [
        b["name"] for b in manifest
        if b["role"] == "primary" and not b.get("downloaded")
    ]
    (EVAL_DIR / "manifest.json").write_text(
        json.dumps(
            {"benchmarks": manifest, "missing_primary": missing},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {(EVAL_DIR / 'manifest.json').relative_to(DATASET.parent)}")

    if missing:
        print("\n" + "!" * 78)
        print(f"MISSING PRIMARY BENCHMARKS: {missing}")
        print("Decontamination cannot cover what is not here.")
        print("!" * 78)


if __name__ == "__main__":
    main()
