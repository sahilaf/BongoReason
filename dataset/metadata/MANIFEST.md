# BongoReason Dataset Manifest

**Current version:** v0.4 · **Built:** 2026-07-31 · **Rebuild:** `python scripts/run_pipeline.py`

v0.4 is produced from the v0.3 master by a five-stage pipeline. Every stage
writes a report to `dataset/metadata/stage0*_*.json`, and
`scripts/validate_dataset.py` gates the result.

> **Revision note (2026-07-31, after Phase 1 literature review).** An earlier
> build of v0.4 misidentified `kawchar85/Bangla-Math` as the BanglaMATH
> benchmark and removed 997 legitimate training records on that basis. It is
> actually BdMO olympiad *training* data with no associated paper. The records
> are restored, the real benchmarks are in place, and lineage-based
> decontamination has been added — which found substantially worse contamination
> that surface matching could not see. See §6.

---

## 1. Directory layout

```
dataset/
├── raw/            untouched source downloads (gitignored — large)
├── eval/           evaluation benchmarks (gitignored — refetchable)
├── interim/        pipeline intermediates (gitignored — regenerable)
├── verified/       bongo_reason_v0.4.json  ← the dataset
├── corrections/    error-correction pairs
├── script_pairs/   dual-script pairs (gitignored — regenerable)
├── splits/         frozen train/val/test id manifests  ← tracked
└── metadata/       this file + per-stage reports
```

## 2. The dataset — `verified/bongo_reason_v0.4.json`

**23,899 records**, down from 30,148 in v0.3.

| Pool | Records | Contents | Used by |
|---|---:|---|---|
| `sft` | 22,837 | question + Bangla reasoning chain + final answer | Phase 5 |
| `rl` | 1,062 | question + final answer only | Phase 6 |

**17,570 records (73.5%) carry a verifiable answer.** Note that the GRPO prompt
pool is not limited to the `rl` pool — GRPO needs only a question and a checkable
answer, so every verifiable record qualifies (15,812 in train).

| Source | Records |
|---|---:|
| `bengali_math_cot` | 14,257 |
| `numina_bn` | 4,643 |
| `somadhan` | 3,994 |
| `bangla_math_bdmo` | 997 |
| `mgsm_bn` | 8 |

### Schema

```json
{
  "id": "somadhan_0",
  "source": "somadhan",
  "pool": "sft",
  "question_bn": "...",              // normalized, original numeral script
  "question_ar_digits": "...",       // same question, Arabic numerals
  "question_bn_digits": "...",       // same question, Bengali numerals
  "solution_bn": "...",              // null for the rl pool
  "solution_ar_digits": "...",
  "solution_bn_digits": "...",
  "final_answer": "৩৬",
  "answer_type": "numeric",
  "verifiable": true,
  "answer_method": "gsm_marker",
  "solution_promoted_from_answer": true,
  "flags": []
}
```

### Answer types

`verifiable` is true exactly for the first five. The tiered verifier is
`scripts/bongo/verify.py`. `expression` and `symbolic` are excluded on purpose —
exact LaTeX equivalence checking is unreliable and a flaky reward is worse than a
narrower one.

| Type | Verifiable | Example |
|---|:--:|---|
| `numeric` | yes | `91`, `৩৬`, `\$50`, `৬০ টাকা` |
| `mcq_option` | yes | `\text{C}`, `\textbf{(B)}\ 2` |
| `fraction` | yes | `\frac{7}{16}` |
| `percent` | yes | `10\%` |
| `bool_bn` | yes | `\text{সঠিক}` |
| `expression` | no | `2x - 2y + z - 1 = 0` |
| `symbolic` | no | `\frac{9\sqrt{3}}{2}` |
| `other` | no | `4:3`, `(2, 3)` |

## 3. Splits — `splits/`

Assigned **after** dedup and decontamination, stratified over
(pool, source, answer_type, verifiable) across 34 strata, keyed on a hash of the
record id so adding records later does not reshuffle existing assignments.

| Split | Records | sft | rl | Verifiable | sha256 (first 16) |
|---|---:|---:|---:|---:|---|
| train | 21,513 | 20,555 | 958 | 15,812 | `0df96787c79ef316` |
| val | 1,193 | 1,141 | 52 | 879 | `bbaf53f692b33de1` |
| test | 1,193 | 1,141 | 52 | 879 | `3caa205e4b1a5df3` |

**Frozen.** Regenerate only when the dataset version changes.

## 4. Dual-script pairs — `script_pairs/script_pairs_v0.4.json`

**22,469 non-degenerate pairs.** 1,430 records are excluded because their
questions contain no digits, making both variants identical — including them
would score a free perfect consistency reward and inflate the metric.

Original numeral script per source:

| Source | Arabic | Bengali | Mixed | No digits |
|---|---:|---:|---:|---:|
| `bengali_math_cot` | 83% | 7% | 3% | 7% |
| `numina_bn` | 83% | 7% | 3% | 7% |
| `somadhan` | 0% | 98% | 0% | 2% |
| `bangla_math_bdmo` | 95% | — | — | 5% |

The split is sharp and it is the empirical basis for the dual-script
contribution — see [`docs/novelty_statement.md`](../../docs/novelty_statement.md).

## 5. Error corrections — `corrections/error_correction_pairs_v0.1.json`

100 records, all `python_rule_based`: `wrong_conversion_factor` (37),
`arithmetic_sign_flip` (32), `formula_reversal` (31). Unchanged by the pipeline.
See open issue O1.

## 6. What the pipeline removed, and why

| Stage | Removed | Reason |
|---|---:|---|
| 1 | 1,000 | `dart_math_bangla` dropped whole — corrupt |
| 1 | 768 | no recoverable final answer |
| 1 | 202 | Chinese text leaked from NuminaMath |
| 2 | 594 | exact (458) + near-duplicate (136) |
| 3 | 9 | direct match against a benchmark |
| 3 | 730 | **lineage: perturbed forms of Bn-MGSM test problems** |
| 3 | 2,946 | **lineage unverifiable: parent benchmark unavailable** |
| | **6,249** | 30,148 → 23,899 |

### Contamination by lineage (the important one)

`distractmath_mgsm` is built by injecting distractors into MGSM problems. Its
738 rows derive from exactly **250 distinct original questions — the entire
Bn-MGSM test set**, verified by matching the raw CSV's `original_question`
column. But because the stored question is the *perturbed* form, only 8 of 738
matched the benchmark textually.

Surface decontamination therefore misses 99% of this contamination. Stage 3 now
reconstructs the perturbed→parent mapping from the raw CSVs and checks lineage.
`DERIVED_SOURCES` in `03_decontaminate.py` declares which sources need it.

`distractmath_msvamp` has the same structure — 2,946 rows from 997 distinct
originals, consistent with the ~1,000-problem MSVAMP test set. Bn-MSVAMP was not
obtainable, so its lineage cannot be cleared and the source is dropped by
default. Override with `--keep-unverifiable-lineage` if you establish that
Bn-MSVAMP is not an evaluation target — but note GanitLLM reports on it, so any
comparison against that work will need it.

### `dart_math_bangla`

Dropped whole. The Bangla columns in the raw CSV are shifted across rows: a row's
`response` begins with the tail of its own `query` and then bleeds into unrelated
problems. Only 573 of 1,000 had solution text, averaging 226 characters against
~1,100 for intact sources, and none had a recoverable answer. `eng_query` /
`eng_response` are intact if re-translation is ever attempted.

### Gold-answer conflicts

Duplicate clusters that disagreed on the gold answer vote: the majority wins, and
a cluster with no majority is kept but demoted to `verifiable: false` and flagged
`answer_conflict`, so it can train SFT without seeding the RL reward with a wrong
target. 53 such records survive.

## 7. Evaluation benchmarks — `eval/`

| Benchmark | Size | Status | Role |
|---|---:|---|---|
| BanglaMATH | 1,763 | downloaded | primary |
| Bn-MGSM | 250 | downloaded | primary (GanitLLM comparison) |
| GSM-Plus-BN | 10,544 | **manual download required** | primary |
| Bn-MSVAMP | ~1,000 | **not located** | needed for GanitLLM comparison |
| BenNumEval | 6 tasks | **gated (403)** | secondary |
| GSM-Plus (EN) | — | downloaded | reference only |

GSM-Plus-BN is on [Mendeley Data](https://data.mendeley.com/datasets/74dscnmrhv/3)
(v3) and needs an interactive download into `dataset/eval/gsm_plus_bn/`. Stage 3
picks up any CSV/JSON placed there.

---

## 8. Open issues

### O1 — Correction set is small and narrow

100 records, 3 rule-based error types, one generator, against a plan that treats
verified localized correction as a contribution. Generate model-realistic
corrections from GRPO rollouts in Phase 6 instead: keep rollouts that reach a
verifiably wrong answer and diff against the gold chain to localize the first
divergent step. Note the literature review found LEMA and LEMMA already do
localized correction in English, so the surviving claim is narrower than planned.

### O2 — GSM-Plus-BN not yet downloaded

The artifact exists (resolved by Phase 1) but requires a manual Mendeley
download. Until it is in place, 3,994 `somadhan` records and 8 `mgsm_bn` records
of GSM8K lineage are unchecked against it.

### O3 — Bn-MSVAMP not located

Blocks both a direct GanitLLM comparison and clearing the 2,946
`distractmath_msvamp` records currently dropped as unverifiable.

### O4 — BenNumEval gated

`ka05ar/BenNumEval` returns 403; request access at its dataset page.

### O5 — Provenance and licensing incomplete

Known: BanglaMATH (CC-BY-4.0), BenNumEval (MIT), GSM-Plus (CC-BY-SA-4.0),
`kawchar85/Bangla-Math` (MIT). Still missing upstream URLs, licenses and
redistribution terms for the remaining training sources. `somadhan` is 3,994
records — 17% of the dataset — and its source paper has not been located.

Note that `bangla_math_bdmo`'s CoT/PoT annotations are **synthetic**, generated
by Qwen-14B and Gemini-pro-002 with only ~300 human-validated, per its dataset
card. Its English chains are discarded here (answer-only, RL pool), but the
answers themselves inherit that provenance.

### O6 — 26% of answers remain unverifiable

`expression`, `symbolic` and `other` are usable as SFT targets but cannot back an
RL reward. Raising this needs a symbolic equivalence checker over LaTeX, which is
a project in itself and was deliberately deferred.

## 9. Raw sources

| File | Rows | Size | Status |
|---|---:|---:|---|
| `bengali_math_cot_raw.csv` | 859,318 | 2.04 GB | in use |
| `numina_bn_subset_raw.csv` | 5,000 | 12.3 MB | in use |
| `somadhan_raw.csv` | 4,001 | 4.8 MB | in use |
| `distractmath_msvamp_raw.csv` | 2,947 | 5.7 MB | **dropped** — unverifiable lineage |
| `bangla_math_bdmo_raw.csv` | 1,455 | 7.7 MB | in use (answer-only) |
| `dart_math_bangla_raw.csv` | 1,000 | 1.0 MB | **dropped** — corrupt |
| `distractmath_mgsm_raw.csv` | 738 | 1.8 MB | **dropped** — Bn-MGSM test lineage |
| `mgsm_bn_train_raw.csv` | 8 | 6 KB | in use (MGSM *train* split, not test) |

## 10. Reproducing

```bash
python scripts/run_pipeline.py
```

Library tests, then stages 0–5 in order, then validation. Roughly a minute. Stage
order is load-bearing: splits must follow decontamination, which must follow
dedup.

Requires `numpy`, `sympy`, `huggingface_hub`, `pyarrow`. All randomness is seeded
and hashing is `blake2b`, not Python's salted builtin `hash()`, so repeated runs
produce byte-identical output.
