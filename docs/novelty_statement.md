# Novelty Statement — BongoReason

**Compiled:** 2026-07-31 · Phase 1 deliverable (M1) · Evidence in
[`literature_review.md`](literature_review.md) and
[`related_work_matrix.md`](related_work_matrix.md)

---

## The statement

> Large language models are trained on text where Bengali numbers appear in two
> scripts — Bengali digits (২৪৩) and Arabic digits (243) — often within the same
> document. A Bangla math model should give the same answer either way. We show
> that it does not, and that this failure is invisible to every existing Bangla
> mathematical benchmark, all of which fix a single numeral script per problem.
>
> We introduce **dual-script consistency**: a paired evaluation protocol that
> measures cross-script answer agreement on the same problem, and a
> reinforcement-learning reward that optimizes it directly. We release
> dual-script variants of existing Bangla math benchmarks, a 25,195-pair training
> corpus, and a 0.6B model trained with the reward, and we show that script
> agreement can be improved without sacrificing accuracy.
>
> We additionally contribute a Bangla error-correction dataset built with a
> **deterministic answer verifier** rather than an LLM judge, making correction
> supervision reproducible and free to generate at scale.

## Why this survives the literature

The novelty rests on one gap that is genuinely open, not four that are mostly
closed.

**Dual-script numeral consistency is unoccupied.**
[ArabicNumBench](https://arxiv.org/html/2602.18776) (2026) is the nearest work
and differs on three axes verified by direct reading: it is evaluation-only with
no training, it reports no cross-script *agreement* rate, and it covers no Indic
script. [NUMCoT](https://arxiv.org/html/2406.02864) covers numerals and units but
not script duality. [GanitLLM](https://arxiv.org/abs/2601.06767) — the closest
system in every other respect — does not mention Bengali digits at all.

The dataset gives this claim empirical footing: our own corpus splits sharply by
script, with `bengali_math_cot` and `numina_bn` at 83% Arabic digits while
`somadhan` is 98% Bengali digits. A model trained across both without a
consistency objective has no reason to be invariant, and nothing currently
measures whether it is.

## What was cut, and why

The plan's four claims do not survive contact with the literature. Three are
withdrawn.

| Original claim | Verdict | Reason |
|---|---|---|
| Verified localized error correction | **Demoted to secondary** | [LEMA](https://arxiv.org/abs/2310.20689) already does mistake-step localization and correction; [LEMMA](https://arxiv.org/abs/2503.17439) and SCoRe extend it. Surviving delta: deterministic verifier instead of GPT-4 judge, in a low-resource language. Real but narrow. |
| Compact Bangla reasoning | **Withdrawn** | A large literature exists (survey [2507.09662](https://arxiv.org/pdf/2507.09662), AALC, S3-CoT, CoT-Valve). GanitLLM already reports 943→193 words in Bengali. |
| Compute-efficient training | **Withdrawn as a claim** | GanitLLM trains a 0.6B model too. LoRA/QLoRA on one GPU is standard practice, not a contribution. Keep as a *property* of the work, stated in one sentence, not as a pillar. |
| Best open sub-1B Bangla model | **Contingent** | GanitLLM-0.6B scores 28.40 Bn-MGSM / 52.40 Bn-MSVAMP. Only claimable by beating those numbers directly. |

## The bar moved

`plan.md` §3 sets the primary success criterion as improving over standard SFT
and over the base model. **That is no longer publishable.** Qwen3-0.6B scores
8.40 on Bn-MGSM; GanitLLM-0.6B scores 28.40. Any Bangla math model at this scale
will now be measured against the latter.

Revised criteria:

1. **Necessary:** measure dual-script agreement on BanglaMATH, GSM-Plus-BN,
   Bn-MGSM and Bn-MSVAMP, and show a substantial gap exists. If models are
   already script-consistent, the contribution collapses and the project should
   be re-scoped again — **test this first, before any training.**
2. **Core:** show the dual-script reward improves agreement without hurting
   accuracy.
3. **Competitive:** be within reach of GanitLLM-0.6B on Bn-MGSM and Bn-MSVAMP.
   Matching while adding script consistency is a defensible result; being far
   behind is not.
4. **Secondary:** show the auto-verified correction data helps over the same
   pipeline without it.

## The cheapest possible falsification

Before writing any training code, run this:

1. Take BanglaMATH and GSM-Plus-BN.
2. Emit each problem twice — all-Bengali digits and all-Arabic digits.
3. Evaluate Qwen3-0.6B and GanitLLM-0.6B on both.
4. Report accuracy on each and the **agreement rate** between them.

If agreement is already above roughly 95%, the central claim is dead and it is
better to know in a day than in month three. If it is meaningfully below that —
which the corpus's script imbalance and the known English-bias findings in
BanglaMATH make likely — the contribution is real, quantified, and the paper has
its opening figure.

`scripts/bongo/normalize.py` already does the conversion in both directions and
`scripts/bongo/verify.py` already grades script-insensitively, so this is an
afternoon of work, not a phase.

## Honest weaknesses

- **Single-axis contribution.** After the cuts this is one idea, not four. That
  is a Findings-tier paper, realistically, and it needs the empirical gap in §
  "cheapest possible falsification" to be substantial.
- **Dual-script may be a tokenizer artifact.** If the effect turns out to be
  entirely explained by Qwen3's tokenizer segmenting Bengali digits poorly, the
  finding is real but the framing shifts from reasoning to tokenization, and
  reviewers will say so. Check tokenization of both scripts early.
- **Search is not exhaustive.** The gap claim rests on targeted searches, not a
  systematic review. Run ACL Anthology full-text and forward-citation searches
  from ArabicNumBench and GanitLLM before submission.
- **Generalization.** One language, one model size, one base family. Applying the
  same protocol to Devanagari or Eastern Arabic numerals would strengthen it
  considerably and is cheap — the conversion is a character map.

## One-line version

*Bangla math models are not invariant to which numeral script a problem is
written in; no benchmark measures this and no training objective targets it; we
provide both, plus a deterministically-verified Bangla correction dataset.*
