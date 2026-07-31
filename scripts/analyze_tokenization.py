"""Compare how the tokenizer segments Bengali versus Arabic numerals.

Any cross-script accuracy gap invites the obvious objection that it is a
tokenizer artifact rather than a reasoning failure. This measures that directly,
so the paper can answer it rather than be asked it.

The key quantity is **digit fertility**: tokens emitted per digit character. A
tokenizer that maps "১২৩" to three tokens but "123" to one is spending three
times the sequence budget on the same number, and any downstream gap is at least
partly mechanical.

    python scripts/analyze_tokenization.py --model Qwen/Qwen3-0.6B
    python scripts/analyze_tokenization.py --model X --benchmark bn_mgsm
"""

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bongo import DATASET, ROOT
from bongo.normalize import BN_DIGITS, to_ar_digits, to_bn_digits

EVAL_DIR = DATASET / "eval_dual_script"
OUT_DIR = ROOT / "results"

DIGIT_RUN = re.compile(r"[0-9]+|[০-৯]+")

# Numbers chosen to probe grouping behaviour: single digits, round numbers,
# multi-digit runs, and decimals.
PROBE_NUMBERS = ["0", "7", "12", "42", "100", "365", "1000", "2026", "12345", "3.14", "1000000"]


def digit_fertility(tokenizer, text):
    """(tokens spanning digit runs, digit characters) for one string."""
    digits = sum(1 for c in text if c.isdigit() or c in BN_DIGITS)
    if not digits:
        return 0, 0
    tokens = 0
    for run in DIGIT_RUN.findall(text):
        tokens += len(tokenizer.encode(run, add_special_tokens=False))
    return tokens, digits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmark", default="all")
    ap.add_argument("--limit", type=int, default=1000)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("transformers is required:  pip install transformers")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"tokenizer: {args.model}   vocab: {tok.vocab_size:,}\n")

    # ---- probe numbers ----
    print("=" * 78)
    print("HOW ARE INDIVIDUAL NUMBERS SEGMENTED?")
    print("=" * 78)
    print(f"{'number':>10}  {'arabic tokens':>14}  {'bengali tokens':>15}   segmentation (bn)")
    ratios = []
    for n in PROBE_NUMBERS:
        bn = to_bn_digits(n)
        ar_ids = tok.encode(n, add_special_tokens=False)
        bn_ids = tok.encode(bn, add_special_tokens=False)
        ratios.append(len(bn_ids) / max(len(ar_ids), 1))
        pieces = [tok.decode([i]) for i in bn_ids]
        print(f"{n:>10}  {len(ar_ids):>14}  {len(bn_ids):>15}   {pieces}")
    print(f"\n  mean bengali/arabic token ratio on probes: {statistics.mean(ratios):.2f}x")

    # ---- unknown / byte fallback ----
    print("\n" + "=" * 78)
    print("ARE BENGALI DIGITS IN THE VOCABULARY AT ALL?")
    print("=" * 78)
    for d_ar, d_bn in zip("0123456789", BN_DIGITS):
        ar_ids = tok.encode(d_ar, add_special_tokens=False)
        bn_ids = tok.encode(d_bn, add_special_tokens=False)
        print(f"  {d_ar} -> {len(ar_ids)} token(s) {ar_ids}    "
              f"{d_bn} -> {len(bn_ids)} token(s) {bn_ids}")

    # ---- on real benchmark text ----
    benchmarks = (["banglamath", "bn_mgsm", "gsm_plus_bn"]
                  if args.benchmark == "all" else [args.benchmark])
    print("\n" + "=" * 78)
    print("ON REAL BENCHMARK QUESTIONS")
    print("=" * 78)
    print(f"{'benchmark':<16}{'n':>7}{'len(ar)':>10}{'len(bn)':>10}{'ratio':>8}"
          f"{'fert(ar)':>10}{'fert(bn)':>10}")

    summary = {}
    for bench in benchmarks:
        path = EVAL_DIR / f"{bench}.jsonl"
        if not path.exists():
            print(f"  {bench}: missing, run scripts/06_build_eval_sets.py")
            continue
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["variants_differ"]:
                    rows.append(r)
                if len(rows) >= args.limit:
                    break
        if not rows:
            continue

        len_ar = len_bn = ft_ar = fd_ar = ft_bn = fd_bn = 0
        for r in rows:
            len_ar += len(tok.encode(r["question_ar_digits"], add_special_tokens=False))
            len_bn += len(tok.encode(r["question_bn_digits"], add_special_tokens=False))
            t, d = digit_fertility(tok, r["question_ar_digits"])
            ft_ar, fd_ar = ft_ar + t, fd_ar + d
            t, d = digit_fertility(tok, r["question_bn_digits"])
            ft_bn, fd_bn = ft_bn + t, fd_bn + d

        n = len(rows)
        fert_ar = ft_ar / max(fd_ar, 1)
        fert_bn = ft_bn / max(fd_bn, 1)
        print(f"{bench:<16}{n:>7,}{len_ar/n:>10.1f}{len_bn/n:>10.1f}"
              f"{len_bn/max(len_ar,1):>8.2f}{fert_ar:>10.2f}{fert_bn:>10.2f}")
        summary[bench] = {
            "n": n,
            "mean_tokens_arabic": len_ar / n,
            "mean_tokens_bengali": len_bn / n,
            "length_ratio": len_bn / max(len_ar, 1),
            "digit_fertility_arabic": fert_ar,
            "digit_fertility_bengali": fert_bn,
            "fertility_ratio": fert_bn / max(fert_ar, 1e-9),
        }

    print("\n  fert = tokens emitted per digit character. 1.00 means one token per digit.")
    if summary:
        worst = max(summary.values(), key=lambda s: s["fertility_ratio"])
        r = worst["fertility_ratio"]
        print("\n  INTERPRETATION:")
        if r >= 2.0:
            print(f"    Bengali digits cost {r:.1f}x more tokens than Arabic.")
            print("    Any accuracy gap is confounded with sequence length. Report this,")
            print("    and control for it (compare at matched token budgets).")
        elif r >= 1.25:
            print(f"    Bengali digits cost {r:.1f}x more tokens — a modest but real asymmetry.")
            print("    Mention it; it weakens but does not invalidate a reasoning claim.")
        else:
            print(f"    Fertility is nearly equal ({r:.2f}x). A gap in accuracy would NOT be")
            print("    explained by tokenization, which materially strengthens the paper.")

    if summary:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"tokenization__{args.model.replace('/', '__')}.json"
        out.write_text(
            json.dumps({"model": args.model, "benchmarks": summary,
                        "probe_ratio_mean": statistics.mean(ratios)}, indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
