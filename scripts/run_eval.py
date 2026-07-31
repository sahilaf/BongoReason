"""Evaluate a model on the dual-script benchmarks and report cross-script agreement.

This is the falsification test for the project's central claim: if a model
answers the same problem identically whether its numerals are Bengali or Arabic,
there is nothing to fix and the contribution collapses. Run this before writing
any training code.

    python scripts/run_eval.py --model Qwen/Qwen3-0.6B --benchmark bn_mgsm
    python scripts/run_eval.py --model dipta007/GanitLLM-0.6B --benchmark all --limit 500
    python scripts/run_eval.py --model X --benchmark all --metrics-only

Results are appended to ``results/<model>/<benchmark>.jsonl`` as they are produced
and completed items are skipped on restart, so a Colab disconnect costs only the
in-flight batch.

The prompt is identical across the two script conditions apart from the question
itself, and contains no digits of its own — otherwise the instruction would leak
a script preference and confound the comparison.
"""

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bongo import DATASET, ROOT
from bongo.answers import canonical, extract_answer
from bongo.verify import verify

EVAL_DIR = DATASET / "eval_dual_script"
RESULTS_DIR = ROOT / "results"

SCRIPTS = {"ar": "question_ar_digits", "bn": "question_bn_digits"}

# Digit-free on purpose — see module docstring.
SYSTEM_PROMPT = (
    "আপনি একজন দক্ষ গণিত শিক্ষক। বাংলায় ধাপে ধাপে সমাধান করুন। "
    "সমাধান শেষে চূড়ান্ত উত্তরটি #### চিহ্নের পরে লিখুন।"
)
USER_TEMPLATE = "{question}\n\nধাপে ধাপে সমাধান করুন এবং চূড়ান্ত উত্তর #### এর পরে লিখুন।"


def load_benchmark(name, limit=None, sample=None, seed=0, verifiable_only=True):
    path = EVAL_DIR / f"{name}.jsonl"
    if not path.exists():
        sys.exit(f"missing {path} — run scripts/06_build_eval_sets.py first")
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            # Only problems that can disagree and can be graded contribute to the
            # agreement metric; keeping the rest would inflate it.
            if verifiable_only and not (r["verifiable"] and r["variants_differ"]):
                continue
            records.append(r)

    if sample and sample < len(records):
        records = stratified_sample(records, sample, seed)
    return records[:limit] if limit else records


def stratified_sample(records, n, seed=0):
    """Sample n records, balanced over the benchmark's own strata.

    GSM-Plus-BN has eight perturbation types and they are the interesting axis —
    a uniform sample would still cover them, but a stratified one guarantees equal
    power per type so per-perturbation breakdowns stay comparable.
    """
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in records:
        groups[r.get("meta", {}).get("perturbation_type", "_")].append(r)

    per = max(1, n // len(groups))
    out = []
    for key in sorted(groups):
        pool = sorted(groups[key], key=lambda r: r["eval_id"])
        rng.shuffle(pool)
        out += pool[:per]
    # Top up if integer division left us short.
    if len(out) < n:
        chosen = {r["eval_id"] for r in out}
        rest = sorted((r for r in records if r["eval_id"] not in chosen),
                      key=lambda r: r["eval_id"])
        rng.shuffle(rest)
        out += rest[: n - len(out)]
    return sorted(out, key=lambda r: r["eval_id"])


def load_done(path):
    done = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["eval_id"], r["script"]))
                except json.JSONDecodeError:
                    continue  # truncated final line from a killed run
    return done


def build_prompt(tokenizer, question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(question=question)},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **({"enable_thinking": False} if "qwen3" in (tokenizer.name_or_path or "").lower() else {}),
        )
    return f"{SYSTEM_PROMPT}\n\n{USER_TEMPLATE.format(question=question)}\n\n"


def generate(args, records):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"  device: {device}")

    out_dir = RESULTS_DIR / args.model.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.benchmark}.jsonl"
    done = load_done(out_path)
    if done:
        print(f"  resuming: {len(done):,} items already complete")

    tasks = [
        (r, script)
        for r in records
        for script in args.scripts
        if (r["eval_id"], script) not in done
    ]
    print(f"  {len(tasks):,} generations to run "
          f"({len(records):,} problems x {len(args.scripts)} scripts)")
    if not tasks:
        return out_path

    # Bengali digits are byte-fallback in Qwen3 (2 tokens each vs 1 for Arabic),
    # so an identical max_new_tokens gives the Bengali condition materially less
    # reasoning room. Truncated chains never emit an answer and score as wrong,
    # which would show up as a "script gap" that is really a budget artifact.
    # Grouping by script lets each condition get its own budget.
    budgets = {s: args.max_new_tokens for s in args.scripts}
    if args.budget == "matched" and "bn" in budgets:
        budgets["bn"] = int(args.max_new_tokens * args.bn_budget_scale)
        print(f"  token budget: ar={budgets.get('ar')}  bn={budgets['bn']} "
              f"(scaled {args.bn_budget_scale}x for byte-fallback digits)")
    else:
        print(f"  token budget: {args.max_new_tokens} for every script (uncontrolled)")

    tasks.sort(key=lambda t: t[1])  # batch by script so each batch has one budget

    started = time.time()
    with out_path.open("a", encoding="utf-8") as sink:
        for i in range(0, len(tasks), args.batch_size):
            batch = tasks[i : i + args.batch_size]
            script_of_batch = batch[0][1]
            batch = [t for t in batch if t[1] == script_of_batch]
            prompts = [build_prompt(tokenizer, r[SCRIPTS[s]]) for r, s in batch]
            enc = tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=args.max_input_tokens,
            ).to(device)

            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=budgets[script_of_batch],
                    do_sample=args.temperature > 0,
                    temperature=args.temperature if args.temperature > 0 else None,
                    top_p=args.top_p if args.temperature > 0 else None,
                    pad_token_id=tokenizer.pad_token_id,
                )

            for (record, script), seq in zip(batch, out):
                new_tokens = seq[enc["input_ids"].shape[1]:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                result = verify(text, record["gold_answer"], record["answer_type"])
                predicted, method = extract_answer(text)
                # Hit the cap without stopping => the chain was cut off, so a
                # missing answer here is a budget failure, not a reasoning one.
                truncated = int(len(new_tokens)) >= budgets[script]
                sink.write(json.dumps({
                    "eval_id": record["eval_id"],
                    "benchmark": record["benchmark"],
                    "script": script,
                    "gold_answer": record["gold_answer"],
                    "answer_type": record["answer_type"],
                    "predicted_raw": predicted,
                    "predicted_canonical": repr(canonical(predicted, record["answer_type"]))
                                           if predicted else None,
                    "extraction_method": method,
                    "correct": result.correct,
                    "reason": result.reason,
                    "output_tokens": int(len(new_tokens)),
                    "token_budget": budgets[script],
                    "truncated": truncated,
                    "output_chars": len(text),
                    "output": text[:2000] if args.save_outputs else None,
                }, ensure_ascii=False) + "\n")
            sink.flush()

            done_n = i + len(batch)
            rate = done_n / max(time.time() - started, 1e-9)
            eta = (len(tasks) - done_n) / max(rate, 1e-9)
            print(f"  {done_n:>6,}/{len(tasks):,}  "
                  f"{rate:.1f} gen/s  eta {eta/60:.1f} min", flush=True)

    return out_path


def report(paths, benchmarks):
    """Per-script accuracy plus the cross-script agreement rate."""
    by_bench = defaultdict(lambda: defaultdict(dict))
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                by_bench[r["benchmark"]][r["eval_id"]][r["script"]] = r

    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)
    print(f"{'benchmark':<14}{'n':>6}{'acc(ar)':>9}{'acc(bn)':>9}{'gap':>7}"
          f"{'agree':>8}{'flip':>7}{'parse(ar)':>11}{'parse(bn)':>11}{'trunc(bn)':>11}")

    summary = {}
    for bench in benchmarks:
        items = by_bench.get(bench, {})
        paired = {k: v for k, v in items.items() if "ar" in v and "bn" in v}
        if not paired:
            continue
        n = len(paired)
        frac = lambda fn: sum(fn(v) for v in paired.values()) / n

        acc_ar = frac(lambda v: v["ar"]["correct"])
        acc_bn = frac(lambda v: v["bn"]["correct"])
        agree = frac(lambda v: v["ar"]["predicted_canonical"] == v["bn"]["predicted_canonical"])
        both = frac(lambda v: v["ar"]["correct"] and v["bn"]["correct"])
        flip = frac(lambda v: v["ar"]["correct"] != v["bn"]["correct"])
        # Diagnostics: an unparseable or truncated generation is a harness
        # failure, not a model judgement, and must not be read as a script effect.
        parse_ar = frac(lambda v: bool(v["ar"].get("extraction_method")))
        parse_bn = frac(lambda v: bool(v["bn"].get("extraction_method")))
        trunc_ar = frac(lambda v: bool(v["ar"].get("truncated")))
        trunc_bn = frac(lambda v: bool(v["bn"].get("truncated")))

        print(f"{bench:<14}{n:>6,}{100*acc_ar:>8.1f}%{100*acc_bn:>8.1f}%"
              f"{100*(acc_ar-acc_bn):>+6.1f}%{100*agree:>7.1f}%{100*flip:>6.1f}%"
              f"{100*parse_ar:>10.0f}%{100*parse_bn:>10.0f}%{100*trunc_bn:>10.0f}%")
        summary[bench] = {
            "n_paired": n, "accuracy_ar": acc_ar, "accuracy_bn": acc_bn,
            "accuracy_gap": acc_ar - acc_bn, "answer_agreement": agree,
            "both_correct": both, "correctness_flip": flip,
            "parseable_ar": parse_ar, "parseable_bn": parse_bn,
            "truncated_ar": trunc_ar, "truncated_bn": trunc_bn,
        }

    if summary:
        n_all = sum(s["n_paired"] for s in summary.values())
        w = lambda k: sum(s[k] * s["n_paired"] for s in summary.values()) / n_all
        print("-" * 88)
        print(f"{'OVERALL':<16}{n_all:>7,}{100*w('accuracy_ar'):>9.1f}%"
              f"{100*w('accuracy_bn'):>9.1f}%{100*(w('accuracy_ar')-w('accuracy_bn')):>+7.1f}%"
              f"{100*w('answer_agreement'):>8.1f}%{100*w('both_correct'):>8.1f}%"
              f"{100*w('correctness_flip'):>6.1f}%")

        print("\n  agree = same final answer under both scripts (correct or not)")
        print("  flip  = correct under one script, wrong under the other")
        print("  parse = share of generations a final answer could be extracted from")
        print("  trunc = share that hit the token cap, so the chain was cut off")

        a, acc = w("answer_agreement"), max(w("accuracy_ar"), w("accuracy_bn"))
        parse = min(w("parseable_ar"), w("parseable_bn"))
        trunc = w("truncated_bn")

        print("\n  READING THE RESULT:")
        # Order matters: a harness failure invalidates the metric entirely, and a
        # model at floor accuracy makes agreement uninformative. Only once both
        # are cleared does the agreement number mean what it looks like.
        blocked = False
        if parse < 0.5:
            blocked = True
            print(f"    !! only {100*parse:.0f}% of generations were parseable.")
            print("    The accuracy numbers measure format compliance, not reasoning.")
            print("    Fix the prompt or raise --max-new-tokens before reading anything else.")
        if trunc > 0.3:
            blocked = True
            print(f"    !! {100*trunc:.0f}% of Bengali generations hit the token cap.")
            print("    Bengali digits are byte-fallback (~2 tokens each), so an equal")
            print("    budget starves that condition. Rerun with --budget matched.")
        if acc < 0.15:
            blocked = True
            print(f"    !! peak accuracy is only {100*acc:.0f}%.")
            print("    Near floor, outputs are close to random and two random answers")
            print("    rarely coincide, so LOW AGREEMENT IS EXPECTED AND MEANS NOTHING.")
            print("    Establish the metric on a model that can actually do the task")
            print("    (GanitLLM-0.6B scores 28.4 on Bn-MGSM) before drawing conclusions.")

        if blocked:
            print("\n    VERDICT: inconclusive — fix the above, then re-read.")
        elif a >= 0.95:
            print("    agreement >= 95% — models are already script-consistent.")
            print("    The dual-script contribution does not survive. Re-scope.")
        elif a >= 0.85:
            print("    agreement 85-95% — a gap exists but is thin.")
            print("    Viable only if the accuracy gap is also material.")
        else:
            print("    agreement < 85%, on a competent model, with parseable outputs")
            print("    and a controlled token budget — the gap is real. This is the figure.")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmark", default="all",
                    help="banglamath | bn_mgsm | gsm_plus_bn | all")
    ap.add_argument("--scripts", default="ar,bn")
    ap.add_argument("--limit", type=int, help="cap problems per benchmark (for a smoke test)")
    ap.add_argument("--sample", type=int,
                    help="stratified subsample per benchmark. The full sweep is "
                         "~20k generations; 1000 gives a +/-3%% CI in a fraction of the time")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--budget", choices=["equal", "matched"], default="matched",
                    help="'matched' scales the Bengali budget to compensate for "
                         "byte-fallback digit tokenization; 'equal' does not")
    ap.add_argument("--bn-budget-scale", type=float, default=1.5,
                    help="Bengali budget multiplier when --budget matched")
    ap.add_argument("--max-input-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--save-outputs", action="store_true",
                    help="store generations for error analysis (large)")
    ap.add_argument("--metrics-only", action="store_true",
                    help="recompute metrics from existing results, no generation")
    args = ap.parse_args()
    args.scripts = args.scripts.split(",")
    sys.stdout.reconfigure(encoding="utf-8")

    all_benchmarks = ["banglamath", "bn_mgsm", "gsm_plus_bn"]
    benchmarks = all_benchmarks if args.benchmark == "all" else [args.benchmark]
    out_dir = RESULTS_DIR / args.model.replace("/", "__")

    paths = []
    for bench in benchmarks:
        if args.metrics_only:
            paths.append(out_dir / f"{bench}.jsonl")
            continue
        records = load_benchmark(bench, args.limit, args.sample, args.seed)
        print(f"\n### {bench}: {len(records):,} gradable problems"
              + (f" (stratified sample of {args.sample}, seed {args.seed})"
                 if args.sample else ""))
        bench_args = argparse.Namespace(**{**vars(args), "benchmark": bench})
        paths.append(generate(bench_args, records))

    summary = report(paths, benchmarks)
    if summary:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "summary.json"
        path.write_text(
            json.dumps({"model": args.model, "results": summary}, indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
