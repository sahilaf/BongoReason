# BongoReason-0.6B — Project Plan

**Compute-Efficient Bangla Mathematical Reasoning with Verified Error Correction and Reinforcement Learning**

---

## 1. Vision

Build an open-source Bangla mathematical reasoning model (≤0.6B parameters) that achieves state-of-the-art performance among open sub-1B models — trainable end-to-end on a single Colab Pro GPU.

Rather than scaling model size or compute, the project wins on:

- **Data quality** — compact, verified, deduplicated reasoning data
- **Error-correction learning** — training on localized, verified mistake corrections
- **Bangla-native reasoning** — not translated-English reasoning artifacts
- **Dual-script consistency** — identical answers across Bangla (০–৯) and Arabic (0–9) numerals
- **Lightweight RL** — GRPO with verifiable rewards (RLVR)

## 2. Core Contributions (Novelty)

1. **Verified localized error correction** — training examples where a specific erroneous step is identified and corrected, with automatic verification.
2. **Dual-script numeral consistency** — first systematic treatment of Bangla/Arabic numeral robustness in math reasoning.
3. **Compact Bangla reasoning** — shorter verified chains outperform long chains at small scale.
4. **Compute-efficient recipe** — full SFT + RL pipeline reproducible on one Colab-class GPU.

## 3. Success Criteria

### Primary (must-hit)

| Criterion | Metric |
|---|---|
| Beat standard SFT on BanglaMath | Accuracy improvement, statistically significant (bootstrap CI) |
| Improve robustness on GSM-Plus-BN | Higher accuracy + lower variance across perturbations |
| Colab-trainable | Every training stage fits ≤ 16 GB VRAM, ≤ 12 h per run |

### Stretch

- Best-performing open Bangla reasoning model under 1B parameters.

## 4. Deliverables

| Category | Items |
|---|---|
| **Research** | Paper, experimental report, ablation study, error analysis |
| **Models** | Base LoRA checkpoint, best SFT model, best RL model, quantized (GGUF/AWQ) inference model |
| **Data** | Compact reasoning dataset, error-correction dataset, dual-script dataset, eval scripts |
| **Open source** | GitHub repo, training code, eval pipeline, docs, model card, data card |

## 5. Tech Stack

- **Base model:** Qwen3-0.6B (optionally a Bangla-native model as comparison)
- **Fine-tuning:** LoRA / QLoRA (Unsloth or PEFT + bitsandbytes)
- **RL:** GRPO via TRL
- **Eval:** custom harness over BanglaMath, GSM-Plus-BN, Bn-MGSM, BenNumEval
- **Compute:** Colab Pro (T4/L4/A100 as available); all runs checkpointed and resumable
- **Tracking:** Weights & Biases (or local CSV logs), fixed seeds, config files per experiment

---

## 6. Current status (as of 2026-07-31)

Phases 1, 2 and 3 are **done**. Phases 2 and 3 were merged — the pipeline had to
exist before the collected data was usable. Details in
[`dataset/metadata/MANIFEST.md`](dataset/metadata/MANIFEST.md) and [`docs/`](docs/).

- **Dataset v0.4 built:** 23,899 records (22,837 SFT / 1,062 RL), 17,570 with a
  verifiable answer. M2's 12,000-problem target is met.
- **Splits frozen** (21,513 / 1,193 / 1,193), 22,469 dual-script pairs built.
- **Serious contamination found by lineage:** all 738 `distractmath_mgsm` records
  are perturbed forms of the 250 Bn-MGSM *test* problems, and only 8 matched
  textually. Both distractor sources are removed. See MANIFEST §6.
- **Phase 1 changes the project's position.** A directly competing system,
  [GanitLLM](https://arxiv.org/abs/2601.06767) (ACL 2026 Findings), already
  published a Bengali Qwen3-0.6B math model trained SFT→GRPO with a
  Bengali-language reward, and released weights, data and code. Three of the four
  planned novelty claims are withdrawn; **dual-script numeral consistency
  survives and is now the core contribution.** See
  [`docs/novelty_statement.md`](docs/novelty_statement.md).

**Next: the falsification test in `docs/novelty_statement.md` — measure whether a
cross-script accuracy gap actually exists, before writing any training code.** If
models are already script-consistent, the contribution collapses and the project
needs re-scoping again. Then Phase 4 baselines, against GanitLLM-0.6B rather than
the untuned base.

## 7. Timeline (14 weeks)

| Phase | Weeks | Focus | Milestone |
|---|---|---|---|
| 1 | 1 | Literature review (done) | M1: Novelty statement ✅ |
| 2+3 | 2–4 | Dataset collection **and** pipeline (merged, done) | M2: 12k verified problems ✅ |
| 4 | 5 | Baselines | M3: Reference scores |
| 5 | 6–7 | Supervised fine-tuning | M4: Best SFT > baseline |
| 6 | 8–9 | Reinforcement learning | M5: Best RL > SFT |
| 7 | 10 | Full evaluation | M6: All results reproducible |
| 8 | 11 | Ablations + error analysis | Component attribution |
| 9 | 12–13 | Paper writing | M7: Submission-ready draft |
| 10 | 14 | Open-source release | Public repo + models |

### Phase 1 — Literature Review (Week 1)

**Goal:** Understand existing work; pin down the exact novelty.

- [x] Bangla reasoning / Bangla NLP papers — **found GanitLLM, a direct competitor**
- [x] LoRA, QLoRA
- [x] GRPO (DeepSeekMath) and RLVR (Tulu 3)
- [x] Mistake-correction / self-correction literature — LEMA, LEMMA, SCoRe
- [x] BanglaMATH and GSM-Plus-BN benchmark papers — both located
- [x] Numeral-script robustness — ArabicNumBench is the only near-neighbour
- [ ] Systematic ACL Anthology full-text + forward-citation search (before submission)

**Deliverables:** [`docs/literature_review.md`](docs/literature_review.md),
[`docs/related_work_matrix.md`](docs/related_work_matrix.md),
[`docs/novelty_statement.md`](docs/novelty_statement.md).

### Phases 2 + 3 — Dataset and Pipeline (Weeks 2–4) — **done**

Merged in practice: the collected data could not be signed off without the
pipeline, because two-thirds of it had no extractable final answer until the
extractor existed.

**Sources collected** — 874,467 raw records across 8 datasets, yielding 26,587
after cleaning. Synthetic generation and human review were not needed to hit the
target and were not done.

**Pipeline built** (`scripts/`, run with `python scripts/run_pipeline.py`):
- [x] Text/numeral normalizer, both script directions (`bongo/normalize.py`)
- [x] Answer extractor — brace-balanced `\boxed{}`, `####`, `উত্তর:` (`bongo/answers.py`)
- [x] Tiered answer verifier, script-insensitive (`bongo/verify.py`)
- [x] Deduplicator — exact + MinHash/LSH near-duplicate (`02_dedup.py`)
- [x] Decontamination against eval benchmarks (`03_decontaminate.py`)
- [x] Dual-script pair builder (`04_build_script_pairs.py`)
- [x] Frozen split assignment (`05_make_splits.py`)
- [x] Dataset validator as a CI gate (`validate_dataset.py`)
- [ ] Equation verifier (step-level arithmetic checking) — **not built**, deferred
      to Phase 6 where it backs the "verified reasoning" reward

**Not done:** compact solution rewriting (belongs in Phase 5, it is a training
target choice) and error-correction expansion (see Phase 6 note).

**Output:** `dataset/verified/bongo_reason_v0.4.json` + frozen splits.
Reproducible byte-for-byte; all hashing is `blake2b`, not Python's salted `hash()`.

### Phase 4 — Baselines (Week 5)

**Evaluate**
- [ ] Base Qwen3-0.6B (zero-shot and few-shot)
- [ ] Optional Bangla-native model

**Measure:** BanglaMath, GSM-Plus-BN, inference speed, memory, output length.

**Deliverable:** baseline report — the reference scores every later experiment is judged against.

### Phase 5 — Supervised Fine-Tuning (Weeks 6–7)

Cumulative experiment ladder (each adds one component):

| Exp | Training data | Tests |
|---|---|---|
| S1 | Answer-only | Does reasoning help at all? |
| S2 | Long reasoning | Long-CoT baseline |
| S3 | Compact reasoning | Compact vs long at 0.6B |
| S4 | S3 + error corrections | Value of correction data |
| S5 | S4 + dual-script | Value of script consistency |

- [ ] Evaluate every experiment on both primary benchmarks
- [ ] Fixed hyperparameters across experiments except the variable under test

**Deliverable:** best SFT model (expected: S4 or S5).

### Phase 6 — Reinforcement Learning (Weeks 8–9)

Train with GRPO starting from the best SFT checkpoint.

**Prompt pool:** 18,272 verifiable training records (`splits/*_train.json`
filtered on `verifiable`). GRPO needs only a question and a checkable gold
answer, so the RL pool includes records whose reference chains were unusable.

**Reward components**
- Correct final answer — `bongo.verify.verify()`, already built and tested
- Verified reasoning steps — needs the step-level equation checker (deferred from Phase 3)
- Reasoning consistency
- Bangla language reward — `bongo.normalize.script_profile()`
- Dual-script consistency — over `script_pairs_v0.4.json` (25,195 pairs)
- Compactness (length penalty)
- Contradiction penalty

> Only `numeric`, `fraction`, `percent`, `mcq_option` and `bool_bn` answers are
> exactly checkable; `expression` and `symbolic` (25% of the data) are SFT-only.
> The verifier collapses numeral script before comparing — without that, correct
> Bengali-numeral generations score as wrong and the dual-script reward inverts.

**Experiments**

| Exp | Reward |
|---|---|
| R1 | Answer-only reward |
| R2 | Multi-reward (answer + verification + language) |
| R3 | Full reward system |

**Error-correction data (issue O1):** generate it here rather than in Phase 2.
Sample from the SFT checkpoint, keep rollouts that reach a *verifiably wrong*
answer, and diff against the gold chain to localize the first divergent step.
These rollouts are needed for GRPO anyway, and model-realistic errors support the
"verified localized correction" claim far better than the current 100
rule-based templates.

**Deliverable:** best RL model.

### Phase 7 — Evaluation (Week 10)

**Primary:** BanglaMath (= `kawchar85/Bangla-Math`, 1,455 problems, downloaded to
`dataset/eval/banglamath_bdmo/`), GSM-Plus-BN (**not yet located** — see §9)
**Secondary:** Bn-MGSM, BenNumEval (gated; access must be requested)

**Metrics:** accuracy, robustness (perturbation variance), script consistency rate, inference speed, memory, output length.

- [ ] All final models evaluated with identical harness, ≥3 seeds where sampling is used
- [ ] Results tables auto-generated from logs
- [ ] Script consistency measured over `script_pairs_v0.4.json`, excluding
      digit-free questions (they would score a free perfect match)

### Phase 8 — Ablations & Error Analysis (Week 11)

**Ablations** — isolate the contribution of:
- [ ] Compact reasoning
- [ ] Error correction data
- [ ] Dual-script training
- [ ] LoRA vs QLoRA
- [ ] RL on top of SFT
- [ ] Individual reward components

**Error analysis** — categorize failures with qualitative examples:
arithmetic, percentages, ratios, units, script confusion, hallucination, wrong reasoning path, wrong final answer despite correct reasoning.

### Phase 9 — Paper Writing (Weeks 12–13)

Sections: Introduction, Related Work, Dataset, Method, Experiments, Results, Ablations, Limitations, Future Work.

- [ ] All figures and tables generated from committed scripts
- [ ] Internal reproducibility check: rerun one full pipeline pass from the repo

### Phase 10 — Open-Source Release (Week 14)

- [ ] Code + training configs
- [ ] Model weights + LoRA adapters (Hugging Face)
- [ ] Evaluation scripts
- [ ] Dataset transformation scripts (respecting source licenses)
- [ ] Documentation, model card, data card

---

## 8. Milestones

| # | Milestone | Success metric | Status |
|---|---|---|---|
| M1 | Literature review done | Clear, written novelty statement | |
| M2 | Dataset complete | 12,000 verified, decontaminated problems | ✅ 26,587 |
| M3 | Baselines done | Reference scores established | |
| M4 | Best SFT model | Beats baseline on both primary benchmarks | |
| M5 | Best RL model | Beats best SFT | |
| M6 | Evaluation complete | All experiments reproducible from repo | |
| M7 | Paper ready | Submission-ready manuscript | |

## 9. Risks & Mitigations

| Risk | Status | Mitigation |
|---|---|---|
| Eval contamination in training data | **Materialized, partly resolved** | BanglaMath was 100% present in training via `bangla_math_bdmo`; all 997 records removed and `validate_dataset.py` now re-checks every build. GSM-Plus-BN remains unmeasured — see below. |
| **GSM-Plus-BN contamination unmeasured** | **Open** | No Bangla release found. 7,686 records (29%) are GSM8K-derived and GSM-Plus perturbs the GSM8K test split. Either obtain the artifact or retain English source questions for English-side matching. Until then it is a stated limitation, not a clean-data claim. |
| Poor Bangla translations | **Materialized, resolved** | `dart_math_bangla` lost sentence alignment in translation and was dropped whole; `bangla_math_bdmo`'s chains were English and are used answer-only. Language checks now run per record. |
| RL instability | Open | Strong SFT first; answer-only reward before full reward system; small KL penalty |
| Compute limits | Open | LoRA/QLoRA primary; checkpoint + resume everything; full FT optional only |
| Weak novelty | Open | Anchor contribution on verified localized correction, dual-script consistency, compact reasoning, compute efficiency |
| Benchmark unavailability | **Partly materialized** | BenNumEval is gated (403). Primary claims rest on BanglaMath; GSM-Plus-BN is contingent on locating the artifact. |
| Correction set too thin to support the headline claim | Open | 100 rule-based examples today. Generate model-realistic corrections from GRPO rollouts in Phase 6. |

## 10. Publication Targets

**Primary:** ACL Findings, EMNLP Findings, COLING, AACL
**Backup:** MathNLP, Low-Resource NLP, and Efficient NLP workshops

## 11. Definition of Success

The project succeeds if it delivers:

1. A reproducible sub-1B Bangla reasoning model.
2. Demonstrated, statistically supported improvements over ordinary SFT.
3. Strong results on BanglaMath and GSM-Plus-BN.
4. A complete open-source training pipeline runnable on Colab Pro.
5. A submission-ready research paper backed by rigorous experiments.
