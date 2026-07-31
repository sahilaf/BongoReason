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
import re
import sys
import time
from collections import defaultdict
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


def load_benchmark(name, limit=None, verifiable_only=True):
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
    return records[:limit] if limit else records


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

    started = time.time()
    with out_path.open("a", encoding="utf-8") as sink:
        for i in range(0, len(tasks), args.batch_size):
            batch = tasks[i : i + args.batch_size]
            prompts = [build_prompt(tokenizer, r[SCRIPTS[s]]) for r, s in batch]
            enc = tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=args.max_input_tokens,
            ).to(device)

            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature if args.temperature > 0 else None,
                    top_p=args.top_p if args.temperature > 0 else None,
                    pad_token_id=tokenizer.pad_token_id,
                )

            for (record, script), seq in zip(batch, out):
                text = tokenizer.decode(
                    seq[enc["input_ids"].shape[1]:], skip_special_tokens=True
                )
                result = verify(text, record["gold_answer"], record["answer_type"])
                predicted, method = extract_answer(text)
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

    print("\n" + "=" * 88)
    print("RESULTS")
    print("=" * 88)
    print(f"{'benchmark':<16}{'n':>7}{'acc(ar)':>10}{'acc(bn)':>10}{'gap':>8}"
          f"{'agree':>9}{'both ok':>9}{'flip':>7}")

    summary = {}
    for bench in benchmarks:
        items = by_bench.get(bench, {})
        paired = {k: v for k, v in items.items() if "ar" in v and "bn" in v}
        if not paired:
            continue
        n = len(paired)
        acc_ar = sum(v["ar"]["correct"] for v in paired.values()) / n
        acc_bn = sum(v["bn"]["correct"] for v in paired.values()) / n
        agree = sum(
            v["ar"]["predicted_canonical"] == v["bn"]["predicted_canonical"]
            for v in paired.values()
        ) / n
        both = sum(v["ar"]["correct"] and v["bn"]["correct"] for v in paired.values()) / n
        flip = sum(
            v["ar"]["correct"] != v["bn"]["correct"] for v in paired.values()
        ) / n
        print(f"{bench:<16}{n:>7,}{100*acc_ar:>9.1f}%{100*acc_bn:>9.1f}%"
              f"{100*(acc_ar-acc_bn):>+7.1f}%{100*agree:>8.1f}%{100*both:>8.1f}%{100*flip:>6.1f}%")
        summary[bench] = {
            "n_paired": n, "accuracy_ar": acc_ar, "accuracy_bn": acc_bn,
            "accuracy_gap": acc_ar - acc_bn, "answer_agreement": agree,
            "both_correct": both, "correctness_flip": flip,
        }

    if summary:
        n_all = sum(s["n_paired"] for s in summary.values())
        w = lambda k: sum(s[k] * s["n_paired"] for s in summary.values()) / n_all
        print("-" * 88)
        print(f"{'OVERALL':<16}{n_all:>7,}{100*w('accuracy_ar'):>9.1f}%"
              f"{100*w('accuracy_bn'):>9.1f}%{100*(w('accuracy_ar')-w('accuracy_bn')):>+7.1f}%"
              f"{100*w('answer_agreement'):>8.1f}%{100*w('both_correct'):>8.1f}%"
              f"{100*w('correctness_flip'):>6.1f}%")

        print("\n  agree   = same final answer under both numeral scripts (correct or not)")
        print("  flip    = correct under one script, wrong under the other")
        print("\n  READING THE RESULT:")
        a = w("answer_agreement")
        if a >= 0.95:
            print("    agreement >= 95% — models are already script-consistent.")
            print("    The dual-script contribution does not survive. Re-scope.")
        elif a >= 0.85:
            print("    agreement 85-95% — a gap exists but is thin.")
            print("    Viable only if the accuracy gap is also material.")
        else:
            print("    agreement < 85% — the gap is real and substantial.")
            print("    This is the paper's opening figure.")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmark", default="all",
                    help="banglamath | bn_mgsm | gsm_plus_bn | all")
    ap.add_argument("--scripts", default="ar,bn")
    ap.add_argument("--limit", type=int, help="cap problems per benchmark (for a smoke test)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=512)
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
        records = load_benchmark(bench, args.limit)
        print(f"\n### {bench}: {len(records):,} gradable problems")
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
