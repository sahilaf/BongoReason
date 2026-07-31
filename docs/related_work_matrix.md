# Related Work Matrix

**Compiled:** 2026-07-31 · Phase 1 deliverable · Sources in [`literature_review.md`](literature_review.md)

## A. Coverage grid — who does what

✅ does it · ➖ partial or incidental · ❌ does not

| Work | Bangla | Math reasoning | ≤1B model | GRPO/RL | Verifiable reward | Error correction | Dual-script numerals | Compact reasoning |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **GanitLLM** (2601.06767) | ✅ | ✅ | ✅ 0.6B | ✅ | ✅ | ❌ | ❌ | ✅ 943→193w |
| DeepSeekMath (2402.03300) | ❌ | ✅ | ❌ 7B | ✅ origin | ✅ | ❌ | ❌ | ❌ |
| Tulu 3 (2411.15124) | ❌ | ➖ | ❌ | ✅ | ✅ RLVR origin | ❌ | ❌ | ❌ |
| LEMA (2310.20689) | ❌ | ✅ | ❌ | ❌ SFT | ➖ GPT-4 judge | ✅ localized | ❌ | ❌ |
| LEMMA (2503.17439) | ❌ | ✅ | ❌ | ❌ SFT | ➖ | ✅ trajectory | ❌ | ❌ |
| SCoRe (ICLR 2025) | ❌ | ✅ | ❌ | ✅ multi-turn | ✅ | ✅ intrinsic | ❌ | ❌ |
| ArabicNumBench (2602.18776) | ❌ | ➖ reading | ❌ eval only | ❌ | ❌ | ❌ | ➖ Arabic, eval only | ❌ |
| NUMCoT (2406.02864) | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ➖ numerals/units | ❌ |
| AALC (2506.20160) | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ length reward |
| S3-CoT (2602.01982) | ❌ | ✅ | ❌ | ❌ | ➖ | ❌ | ❌ | ✅ |
| BanglaMATH (2510.12836) | ✅ | ✅ | ❌ eval only | ❌ | ❌ | ❌ | ❌ | ❌ |
| GSM-Plus-BN (2607.13248) | ✅ | ✅ | ❌ eval only | ❌ | ❌ | ❌ | ➖ digit expansion | ❌ |
| LoRA / QLoRA | ❌ | ❌ | ➖ enables | ❌ | ❌ | ❌ | ❌ | ❌ |
| **BongoReason (this work)** | ✅ | ✅ | ✅ 0.6B | ✅ | ✅ | ✅ auto-verified | ✅ **train + eval** | ➖ |

**Only one column has no ✅ above our row: dual-script numeral consistency as a
training signal.** Everything else is occupied, and GanitLLM occupies most of it
in the same language at the same model size.

## B. The competitor, in detail

GanitLLM is the paper this work will be reviewed against. Direct comparison:

| Dimension | GanitLLM | BongoReason (planned) |
|---|---|---|
| Venue | ACL 2026 Findings (published) | target ACL/EMNLP Findings |
| Base model | Qwen3 (0.6B–32B sweep) | Qwen3-0.6B |
| Flagship | GanitLLM-4B | 0.6B only |
| Method | SFT → Curriculum-GRPO, difficulty-aware sampling | SFT → GRPO |
| Rewards | format, correctness (+Bengali bonus), Bengali-token ratio | + dual-script consistency, + correction |
| Training data | *Ganit*, difficulty-tagged, released | 26.6k records, 8 sources |
| Decontamination | MinHash vs MGSM/MSVAMP @ 50% | MinHash @ 80–90%, benchmarks TBD |
| Eval | Bn-MGSM, Bn-MSVAMP | BanglaMATH, GSM-Plus-BN (+ must add Bn-MGSM/MSVAMP) |
| Numeral script | **not addressed** | core contribution |
| Error correction | not addressed | core contribution |
| Released | weights + data + code | planned |

**Reference numbers to beat** (from the GanitLLM project site):

| Model | Bn-MGSM | Bn-MSVAMP |
|---|---:|---:|
| Qwen3-0.6B base | 8.40 | 12.20 |
| GanitLLM-0.6B | 28.40 | 52.40 |

Beating Qwen3-0.6B is no longer a result. GanitLLM-0.6B is the bar.

## C. Benchmark inventory

| Benchmark | Size | Content | Access | Role here |
|---|---:|---|---|---|
| BanglaMATH | 1,700 | grades 6–8 school workbook problems | `github.com/BanglaMATH` | primary |
| GSM-Plus-BN | 10,544 (9k eval) | GSM-Plus perturbations, human-translated | [Mendeley v3](https://data.mendeley.com/datasets/74dscnmrhv/3) | primary |
| Bn-MGSM | — | MGSM Bengali split | via GanitLLM | **add** — needed for direct comparison |
| Bn-MSVAMP | — | MSVAMP Bengali split | via GanitLLM | **add** — needed for direct comparison |
| BenNumEval | ~6 tasks | Bengali numerical reasoning | gated, request access | secondary |
| `kawchar85/Bangla-Math` | 1,455 | BdMO olympiad, synthetic CoT/PoT | HF, open | **training data, not a benchmark** |

## D. Positioning claims and their support

| Claim | Supportable? | Why |
|---|---|---|
| "First dual-script consistency treatment for Bangla math" | **Yes** | ArabicNumBench is Arabic-only, eval-only, no agreement metric, no Indic scripts |
| "Dual-script consistency as a training reward" | **Yes** | no prior work found using script agreement as a training signal in any language |
| "First Bangla error-correction dataset with automatic verification" | **Qualified yes** | LEMA/LEMMA precede it in English with LLM-judge correction; the delta is the deterministic verifier and the language |
| "Best open sub-1B Bangla reasoning model" | **Only if we beat GanitLLM-0.6B** | 28.40 / 52.40 is public |
| "Compact Bangla reasoning" | **No** | GanitLLM already reports 943→193 words |
| "Compute-efficient Bangla reasoning recipe" | **No** | GanitLLM trains a 0.6B too; efficiency alone is not a contribution |

## E. Gaps in this review

- Systematic ACL Anthology full-text search not yet run
- No forward citation search from ArabicNumBench, NUMCoT, or GanitLLM
- PATIGONIT22K (2607.22859) retrieved but not read — may be a training-data source
- SOMADHAN's own paper not located; it is 3,994 records of our training data and
  its provenance and license are unverified
- MGSM and MSVAMP primary sources not retrieved
