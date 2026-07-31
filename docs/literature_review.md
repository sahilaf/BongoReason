# Literature Review — BongoReason-0.6B

**Compiled:** 2026-07-31 · Phase 1 deliverable

Every reference below was retrieved and read during compilation. Where a work is
cited only because another paper cites it, that is marked **[secondary]** — those
need first-hand checking before they go in a submission. Nothing here is cited
from memory.

---

## 0. The headline finding

**A directly competing system already exists and is published.**
[GanitLLM](https://arxiv.org/abs/2601.06767) (Dipta, Mahbub & Najjar, ACL 2026
Findings, January 2026) is a Bengali mathematical reasoning model trained with
SFT followed by GRPO, with a Bengali-language reward, MinHash decontamination,
and a released dataset, code, and weights.

It includes **GanitLLM-0.6B**, built on the same Qwen3-0.6B base this project
proposes:

| Model | Bn-MGSM | Bn-MSVAMP |
|---|---:|---:|
| Qwen3-0.6B (base) | 8.40 | 12.20 |
| **GanitLLM-0.6B** | **28.40** | **52.40** |

Released at
[HuggingFace collection](https://huggingface.co/collections/dipta007/ganitllm-acl-2026-findings),
dataset at [dipta007/Ganit](https://huggingface.co/datasets/dipta007/Ganit),
code at [github.com/dipta007/GanitLLM](https://github.com/dipta007/GanitLLM).

This removes two of the four novelty claims in `plan.md` outright and forces a
re-scope. See [`novelty_statement.md`](novelty_statement.md).

---

## 1. Bangla mathematical reasoning

### 1.1 Models

**GanitLLM** (arXiv [2601.06767](https://arxiv.org/abs/2601.06767)) — Qwen3-4B
flagship plus a 0.6B–32B sweep. Contributions: the *Ganit* corpus with automatic
difficulty tags derived from evaluator-model performance; Curriculum-GRPO
combining multi-stage training (SFT → GRPO) with difficulty-aware sampling.

Reward design (three components, total 0–4):
- format reward (0–1) — output structure
- correctness reward (0–2) — accuracy, with a bonus for reasoning in Bengali
- Bengali reasoning reward (0–1) — requires ≥80% Bengali tokens

Reported effects: Bengali reasoning tokens 14% → >88%; solution length 943 → 193
words; +8 Bn-MGSM and +6 Bn-MSVAMP over base at 4B.

Decontamination: MinHash against MGSM and MSVAMP, removing training instances
above 50% similarity.

Stated limitations: Bengali-only; difficulty estimated by proxy signal; the
language-fidelity reward is a character-percentage heuristic that "may
incorrectly penalize valid outputs that mix languages."

**Crucially, GanitLLM does not discuss Bengali numeral scripts or digits at all.**
Verified by direct reading of the paper HTML.

### 1.2 Benchmarks

**BanglaMATH** (arXiv [2510.12836](https://arxiv.org/abs/2510.12836);
[MathNLP 2025](https://aclanthology.org/2025.mathnlp-main.10/)) — Prama, Danforth
& Dodds. 1,700 Bangla math word problems from **elementary school workbooks**,
grades 6–8, covering arithmetic, algebra, geometry and logical reasoning.
Annotated with grade level, number of reasoning steps, explanations, and digit
count. Only Gemini 2.5 Flash and DeepSeek V3 reach ≥80% across all three grades;
the paper reports significant robustness and language-bias issues. Repository
referenced as `github.com/BanglaMATH`.

> **This is not the same artifact as `kawchar85/Bangla-Math`.** See §5.

**GSM-Plus-BN** (arXiv [2607.13248](https://arxiv.org/abs/2607.13248)) — Paul,
Mayouree, Karim, Nath & Kundu, July 2026. Bengali adaptation of English GSM-Plus,
human-translated and verified by six Bengali translators with NLP backgrounds.
10,544 total instances from 1,318 seed questions; the evaluation uses 9,000
(1,000 seed + 8,000 perturbed). Eight perturbation types: numerical substitution,
digit expansion, integer-decimal-fraction conversion, adding operation, reversing
operation, problem understanding, distraction insertion, critical thinking.

**Released on Mendeley Data:**
<https://data.mendeley.com/datasets/74dscnmrhv/3> (v3). No GitHub or HuggingFace
mirror. This resolves open issue O2 — the artifact exists and is downloadable.

Evaluated Qwen3-32B, Llama-3.1-8B, Llama-3.3-70B, Llama-4-Scout-17B,
GPT-OSS-120B and GPT-OSS-20B. GPT-OSS-20B reaches 96.08% on seed questions under
standard prompting; larger models are more robust to perturbation; a gap to
English persists across all models.

**Bn-MGSM / Bn-MSVAMP** — Bengali splits of MGSM (Shi et al. 2023) **[secondary]**
and MSVAMP (Chen et al. 2023) **[secondary]**, used as GanitLLM's evaluation
suite. GanitLLM reports MGSM is 77.5% easy problems with only 2.5% olympiad-level,
which is worth knowing before adopting it as a headline benchmark.

**PATIGONIT22K** (arXiv [2607.22859](https://arxiv.org/abs/2607.22859)) — a
Bengali math word problem dataset, July 2026. Retrieved in search but not read in
full; worth checking as a training-data source since it postdates the current
collection.

**BenNumEval** ([ka05ar/BenNumEval](https://huggingface.co/datasets/ka05ar/BenNumEval))
— Bengali numerical reasoning, six tasks. **Gated**; access must be requested.

### 1.3 Bangla NLP infrastructure

**BanglaBERT** (csebuetnlp, NAACL 2022 Findings) and **BanglaT5/BanglaNLG** —
the standard pretrained encoders/generators for Bangla, useful as citations for
the low-resource framing rather than as components here.

---

## 2. Parameter-efficient fine-tuning

**LoRA** (arXiv [2106.09685](https://arxiv.org/abs/2106.09685), Hu et al.) —
freezes pretrained weights and injects trainable rank-decomposition matrices into
each Transformer layer. Reports up to 10,000× fewer trainable parameters and 3×
lower GPU memory versus full fine-tuning of GPT-3 175B.

**QLoRA** (arXiv [2305.14314](https://arxiv.org/abs/2305.14314), Dettmers,
Pagnoni, Holtzman & Zettlemoyer, 2023) — 4-bit NormalFloat quantization, double
quantization, and paged optimizers, enabling finetuning of quantized models with
LoRA adapters.

Both are settled infrastructure. They support the compute-efficiency framing but
are not themselves a contribution.

---

## 3. Reinforcement learning for reasoning

**GRPO / DeepSeekMath** (arXiv [2402.03300](https://arxiv.org/abs/2402.03300),
Shao et al., 2024) — Group Relative Policy Optimization, a PPO variant that drops
the learned value function and instead normalizes rewards within a sampled group,
cutting memory from three model copies to two. DeepSeekMath 7B reaches 51.7% on
MATH without external tools or voting.

**RLVR** — the term originates jointly with DeepSeekMath and **Tulu 3**
(arXiv [2411.15124](https://arxiv.org/abs/2411.15124), Lambert et al., 2024).
The RLHF objective is retained but the learned reward model is replaced by a
deterministic verification function returning binary reward. Known weakness: rule
based rewards do not scale to tasks whose correctness is not mechanically
checkable, and hand-designed reward functions are vulnerable to reward hacking.

This limitation is directly load-bearing for this project — 25% of our answers
(`expression`, `symbolic`) are not mechanically checkable, which is why they are
SFT-only.

---

## 4. Learning from mistakes

This area is more developed than `plan.md` assumes.

**LEMA** (arXiv [2310.20689](https://arxiv.org/abs/2310.20689), "Learning From
Mistakes Makes LLM Better Reasoner") — collects incorrect reasoning paths, then
uses GPT-4 as a *corrector* to identify the mistaken step, explain why it is
wrong, correct it, and produce the final answer. Fine-tunes on the resulting
mistake–correction pairs. **This is essentially the "localized correction" idea,
already published for English.**

**LEMMA** (arXiv [2503.17439](https://arxiv.org/abs/2503.17439), ACL Findings
2025) — constructs self-correction data as a concatenation of a bad trajectory, a
reflection phrase pinpointing the error, and a correct trajectory.

**SCoRe** (ICLR 2025) — multi-turn RL for intrinsic self-correction; reports
+15.6 points on self-correction for MATH and +9.1 on HumanEval. First method to
achieve significantly positive *intrinsic* self-correction.

**Self-Error-Instruct** (arXiv [2505.22591](https://arxiv.org/abs/2505.22591)) —
generalizes from observed errors to generate targeted training data.

**Critical survey** — "When Can LLMs Actually Correct Their Own Mistakes?"
(TACL) — argues much reported self-correction gain does not survive scrutiny.
Read this before framing any self-correction claim.

**Implication:** "verified localized error correction" is not novel as a
mechanism. The remaining delta is the *verifier*: LEMA uses GPT-4 as judge,
whereas an automatic answer verifier gives a deterministic, reproducible,
zero-cost correctness signal. That distinction is defensible but much narrower
than the plan's framing.

---

## 5. Numeral script and dual-script reasoning

**This is where the surviving novelty is.**

**ArabicNumBench** (arXiv [2602.18776](https://arxiv.org/html/2602.18776),
Alhumud, Alhammadi & Khan, 2026) — the closest prior work. Evaluates LLM handling
of Eastern Arabic-Indic versus Western Arabic numerals, and tracks a *Format
Preservation* metric (does the model answer in the expected numeral script).

Confirmed by direct reading:
- it is **evaluation only** — no models are trained or fine-tuned
- it reports **no cross-script agreement or consistency rate**
- it makes **no mention of Bengali, Devanagari, or any Indic script**

**NUMCoT** (arXiv [2406.02864](https://arxiv.org/html/2406.02864)) — numerals and
units of measurement in chain-of-thought reasoning; finds LLMs handle
number↔English text conversion robustly but number↔Chinese text poorly.

**Gap:** no work found that (a) treats Bengali versus Arabic numerals as a
consistency property, (b) measures cross-script *agreement* on the same problem,
or (c) uses script consistency as a *training* signal. Searches covering numeral
script robustness, Indic digits, and Bengali numerals returned nothing on point.

Caveat: absence of evidence from a handful of searches is not proof of absence.
Before submission, run a proper systematic search (ACL Anthology full-text,
Semantic Scholar citation graph forward from ArabicNumBench and NUMCoT).

---

## 6. Compact reasoning

Heavily worked and **not novel**.

- **Survey:** "Towards Concise and Adaptive Thinking in Large Reasoning Models"
  (arXiv [2507.09662](https://arxiv.org/pdf/2507.09662))
- **AALC** (arXiv [2506.20160](https://arxiv.org/html/2506.20160)) — adaptive
  accuracy-length control, RL with length-aware reward
- **S3-CoT** (arXiv [2602.01982](https://arxiv.org/pdf/2602.01982)) —
  self-sampled succinct reasoning
- **Focused CoT** (arXiv [2511.22176](https://arxiv.org/pdf/2511.22176))
- CoT-Valve, TokenSkip **[secondary]** — SFT on variable-length CoT
- Coconut, SoftCoT **[secondary]** — latent reasoning

Established families: RL with length rewards, SFT on variable-length CoT,
prompt-level budgets, and latent reasoning. GanitLLM already demonstrates the
length reduction in Bengali specifically (943 → 193 words).

---

## 7. Benchmark identity — a correction

`dataset/raw/bangla_math_bdmo_raw.csv` is byte-identical in structure to
[`kawchar85/Bangla-Math`](https://huggingface.co/datasets/kawchar85/Bangla-Math)
(1,455 rows, same six columns). That dataset's README describes it as:

- **Bangladesh Mathematical Olympiad (BdMO)** problems, rounds 2005–2023,
  extracted from PDFs by OCR, plus ~200 problems from a Kaggle competition
- CoT and PoT annotations **synthetically generated** by Qwen-14B and
  Gemini-pro-002, with ~300 human-validated
- explicitly **a training dataset, not an evaluation benchmark**
- **no associated paper**

It is therefore *not* BanglaMATH. BanglaMATH is Prama et al.'s 1,700 grade-6–8
school-workbook problems (§1.2). The names collide; the artifacts do not.

Consequence for this project: the stage-3 decontamination that removed 997
`bangla_math_bdmo` records matched them against a copy of themselves, not against
an evaluation benchmark. That removal was unfounded and is being reversed; real
decontamination against BanglaMATH and GSM-Plus-BN has not yet run.

---

## 8. What this means for the plan

1. **Sub-1B Bangla reasoning is taken.** GanitLLM-0.6B is published, released, and
   has real numbers. The stretch goal in `plan.md` §3 is no longer available as
   stated.
2. **Compact reasoning and a Bangla-language reward are taken**, both by GanitLLM.
3. **Localized error correction is taken** in English (LEMA, LEMMA, SCoRe); only
   the deterministic-verifier variant is open.
4. **Dual-script numeral consistency is open** and is now the strongest claim.
5. **GanitLLM-0.6B is the baseline to beat**, not Qwen3-0.6B. Beating an
   untuned base model is no longer a publishable result.
6. **Bn-MGSM and Bn-MSVAMP** should be added to the evaluation suite, because
   that is what the competing system reports and comparisons need to be direct.

See [`novelty_statement.md`](novelty_statement.md) for the re-scoped position and
[`related_work_matrix.md`](related_work_matrix.md) for the coverage grid.

---

## 9. Verification status

Read in full or in substantial part: GanitLLM (abstract + HTML + project site),
GSM-Plus-BN (abstract + HTML), BanglaMATH (ACL abstract page), ArabicNumBench
(HTML), `kawchar85/Bangla-Math` (dataset card).

Confirmed via search result summaries and abstract pages only — **verify before
citing in the paper**: LoRA, QLoRA, DeepSeekMath, Tulu 3, LEMA, LEMMA, SCoRe,
Self-Error-Instruct, the TACL self-correction survey, NUMCoT, the concise-thinking
survey, AALC, S3-CoT, PATIGONIT22K, BanglaBERT.

Marked **[secondary]** and not independently retrieved: MGSM (Shi et al. 2023),
MSVAMP (Chen et al. 2023), CoT-Valve, TokenSkip, Coconut, SoftCoT, SOMADHAN
(Paul et al. 2025).

Not yet done: systematic ACL Anthology full-text search; forward citation search
from ArabicNumBench and GanitLLM; a check of whether anyone has cited GanitLLM
since January 2026.
