"""Run the full data pipeline in order, then validate.

    python scripts/run_pipeline.py            # every stage
    python scripts/run_pipeline.py --from 2   # resume at stage 2
    python scripts/run_pipeline.py --skip-fetch

Stage order is not negotiable: splits must be assigned after decontamination,
and decontamination after dedup, or earlier stages invalidate later ones.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

STAGES = [
    (0, "00_fetch_eval.py", "fetch evaluation benchmarks"),
    (1, "01_unify_schema.py", "unify schema, extract answers, assign pools"),
    (2, "02_dedup.py", "exact + near-duplicate removal"),
    (3, "03_decontaminate.py", "remove eval-set contamination"),
    (4, "04_build_script_pairs.py", "build dual-script pairs"),
    (5, "05_make_splits.py", "assign frozen splits"),
    (6, "06_build_eval_sets.py", "build dual-script evaluation sets"),
]

TESTS = ["test_bongo.py", "test_eval_metrics.py"]


def run(script):
    result = subprocess.run([sys.executable, str(HERE / script)])
    return result.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=0)
    ap.add_argument("--skip-fetch", action="store_true",
                    help="skip stage 0 (needs network)")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()

    if not args.skip_tests:
        for test in TESTS:
            print("=" * 78)
            print(test)
            print("=" * 78)
            if run(test) != 0:
                sys.exit(f"{test} failed — fix before running the pipeline")

    for number, script, description in STAGES:
        if number < args.start or (number == 0 and args.skip_fetch):
            print(f"\n--- skipping stage {number}: {description}")
            continue
        print("\n" + "#" * 78)
        print(f"# stage {number}: {description}")
        print("#" * 78)
        started = time.time()
        code = run(script)
        if code != 0:
            sys.exit(f"\nstage {number} ({script}) failed with exit code {code}")
        print(f"--- stage {number} finished in {time.time()-started:.1f}s")

    print("\n" + "#" * 78)
    print("# validation")
    print("#" * 78)
    sys.exit(run("validate_dataset.py"))


if __name__ == "__main__":
    main()
