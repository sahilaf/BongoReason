# BongoReason-0.6B

Compute-efficient Bangla mathematical reasoning with verified error correction
and reinforcement learning. See [`plan.md`](plan.md) for the project plan and
[`dataset/metadata/MANIFEST.md`](dataset/metadata/MANIFEST.md) for the data card.

## Status

Dataset **v0.4** is built: 23,899 records (22,837 SFT / 1,062 RL), 17,570 with a
verifiable answer, splits frozen.

Phase 1 is done and it re-scoped the project — see
[`docs/novelty_statement.md`](docs/novelty_statement.md).
[GanitLLM](https://arxiv.org/abs/2601.06767) (ACL 2026 Findings) already released
a Bengali Qwen3-0.6B math model trained SFT→GRPO. **Dual-script numeral
consistency is the surviving contribution.**

The falsification test is built and ready to run: 9,993 gradable dual-script
problem pairs across three benchmarks. See
[`docs/colab_quickstart.md`](docs/colab_quickstart.md).

**Run it:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sahilaf/BongoReason/blob/main/notebooks/01_script_gap_experiment.ipynb)
or locally:

```bash
python scripts/run_eval.py --model Qwen/Qwen3-0.6B --benchmark bn_mgsm --limit 20
```

## Data

The built dataset lives on Hugging Face at
[`sahilfarib/bongo-reason`](https://huggingface.co/datasets/sahilfarib/bongo-reason)
— it is 165 MB, over GitHub's per-file limit. It is **private** until source
licensing is resolved; see the Licensing section of its dataset card.

Evaluation benchmarks are deliberately not redistributed in either place. Fetch
them from their original sources with `scripts/00_fetch_eval.py`.

## Quick start

```bash
python scripts/run_pipeline.py
```

Runs the library tests, then all pipeline stages, then validation — about a
minute. Requires `numpy`, `sympy`, `huggingface_hub`, `pyarrow`.

GSM-Plus-BN needs a manual download from
[Mendeley](https://data.mendeley.com/datasets/74dscnmrhv/3) into
`dataset/eval/gsm_plus_bn/`; stage 3 picks it up automatically.

## Layout

```
docs/
├── literature_review.md        Phase 1 — verified sources, competitor analysis
├── related_work_matrix.md      coverage grid + benchmark inventory
├── novelty_statement.md        M1 — the re-scoped claim and how to falsify it
└── colab_quickstart.md         how to run the script-gap experiment

scripts/
├── bongo/                      shared library
│   ├── normalize.py            numeral conversion (both directions), script detection
│   ├── answers.py              \boxed extraction, answer typing
│   ├── verify.py               tiered verifier — the RL reward
│   └── dedup.py                MinHash + LSH
├── 00_fetch_eval.py            download evaluation benchmarks
├── 01_unify_schema.py          extract answers, assign sft/rl pools
├── 02_dedup.py                 exact + near-duplicate removal
├── 03_decontaminate.py         remove eval-set overlap
├── 04_build_script_pairs.py    dual-script pairs
├── 05_make_splits.py           frozen train/val/test
├── 06_build_eval_sets.py       dual-script evaluation sets
├── run_eval.py                 evaluate a model, report cross-script agreement
├── analyze_tokenization.py     Bengali vs Arabic numeral tokenization
├── validate_dataset.py         CI gate over the built dataset
├── test_bongo.py               library tests
├── test_eval_metrics.py        agreement-metric tests
└── run_pipeline.py             run everything in order

dataset/
├── verified/bongo_reason_v0.4.json    the dataset
├── splits/                            frozen id manifests (tracked)
├── metadata/                          data card + per-stage reports
└── raw/ eval/ interim/ script_pairs/  gitignored, regenerable
```

Stage order is load-bearing: splits must follow decontamination, which must
follow dedup. Running them out of order silently invalidates the result.

## Using the verifier

```python
from bongo.verify import verify

verify(model_output, gold_answer, answer_type)  # -> Result(correct=..., reason=...)
```

It extracts the final answer from a full generation the same way the training
data was built, and collapses numeral script before comparing — a Bengali-digit
answer matches an Arabic-digit gold and vice versa.

Only `numeric`, `fraction`, `percent`, `mcq_option` and `bool_bn` are exactly
checkable. `expression` and `symbolic` answers (25% of the data) are SFT-only.

## Known gaps

Tracked as O1–O5 in the manifest. The two that matter: GSM-Plus-BN contamination
is unmeasured because no Bangla release was found, and the error-correction set
is still 100 rule-based examples against a plan that treats verified localized
correction as the headline contribution.
