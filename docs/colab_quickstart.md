# Colab Quickstart — the script-gap experiment

This runs the falsification test in
[`novelty_statement.md`](novelty_statement.md): does a Bangla math model give the
same answer when a problem's numerals are Bengali versus Arabic?

Everything below fits a Colab Pro T4. Expect roughly 1–3 hours for the full
sweep, less with `--limit`.

---

## 1. Setup

```bash
!git clone <your-repo-url> BanglaMath && cd BanglaMath
!pip install -q transformers accelerate pyarrow sympy
```

The evaluation sets are built from the benchmarks in `dataset/eval/`. If you
cloned without them (they are gitignored), rebuild:

```bash
!python scripts/00_fetch_eval.py
!python scripts/06_build_eval_sets.py
```

GSM-Plus-BN needs a manual download from
[Mendeley](https://data.mendeley.com/datasets/74dscnmrhv/3) into
`dataset/eval/gsm_plus_bn/`. The other benchmarks fetch automatically.

Mount Drive first if you want results to survive a disconnect:

```python
from google.colab import drive; drive.mount('/content/drive')
# then symlink results/ into Drive
```

## 2. Smoke test first

Always do this before the long run — it catches a broken prompt or a tokenizer
issue in two minutes instead of two hours.

```bash
!python scripts/run_eval.py --model Qwen/Qwen3-0.6B --benchmark bn_mgsm --limit 20 --save-outputs
```

Check that `extraction_method` is mostly not null in
`results/Qwen__Qwen3-0.6B/bn_mgsm.jsonl`. If the model never emits a parseable
answer, the accuracy number is measuring format compliance, not reasoning — fix
the prompt in `run_eval.py` before continuing.

## 3. Tokenization analysis

Cheap, and it determines how you have to frame any gap you find.

```bash
!python scripts/analyze_tokenization.py --model Qwen/Qwen3-0.6B
```

If Bengali digits cost ≥2× the tokens of Arabic digits, any accuracy gap is
confounded with sequence length and must be reported as such.

## 4. The main run

```bash
!python scripts/run_eval.py --model Qwen/Qwen3-0.6B --benchmark all --batch-size 32
!python scripts/run_eval.py --model dipta007/GanitLLM-0.6B --benchmark all --batch-size 32
```

Both models matter. Qwen3-0.6B is the untuned base; GanitLLM-0.6B is the
published competitor and the real bar. If the base model is script-inconsistent
but GanitLLM already is not, the contribution is already solved and you need to
know that.

Interrupted? Rerun the same command — completed items are skipped.

Recompute metrics without regenerating:

```bash
!python scripts/run_eval.py --model Qwen/Qwen3-0.6B --benchmark all --metrics-only
```

## 5. Reading the output

```
benchmark             n   acc(ar)   acc(bn)     gap    agree  both ok   flip
bn_mgsm             245     ...       ...       ...     ...     ...     ...
```

- **agree** — the model gave the *same* final answer under both scripts, right or
  wrong. This is the headline number.
- **flip** — correct under one script, wrong under the other. The clearest
  evidence of script sensitivity.
- **gap** — accuracy difference. Positive means Arabic digits are easier.

Decision rule, also printed by the script:

| Agreement | Verdict |
|---|---|
| ≥ 95% | Contribution does not survive. Re-scope. |
| 85–95% | Thin. Viable only if the accuracy gap is also material. |
| < 85% | Real and substantial. This is the paper's opening figure. |

## 6. What the eval sets contain

Only problems that can actually disagree and can actually be graded — a pair
whose two variants are identical would score a free match and inflate agreement.

| Benchmark | Problems | Gradable pairs | Native script |
|---|---:|---:|---|
| BanglaMATH | 1,703 | 675 | 49% Bengali, 27% Arabic |
| Bn-MGSM | 250 | 245 | 98% Arabic |
| GSM-Plus-BN | 10,544 | 9,073 | 98% Bengali |
| **Total** | | **9,993** | |

The two large benchmarks sit at opposite ends of the script spectrum, which is
useful: Bn-MGSM tests degradation when you convert *to* Bengali digits, and
GSM-Plus-BN tests degradation when you convert *to* Arabic. A one-sided effect
would be as interesting as a symmetric one.

Note BanglaMATH is only 43% verifiable — many of its `Answer` values are prose
definitions rather than numbers, so it contributes least despite being a primary
benchmark.

## 7. If the result is positive

Next experiment is SFT with script augmentation, which needs no RL:

- train split: `dataset/splits/sft_train.json` (20,555 records)
- augmentation source: `dataset/script_pairs/script_pairs_v0.4.json` (22,469 pairs)

Train once without augmentation and once with, hold everything else fixed, and
re-run this same evaluation. That contrast is the paper's core result, and it is
two LoRA runs rather than a GRPO campaign.
