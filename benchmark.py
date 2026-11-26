"""Benchmark (optional) - score the trained model on public multiple-choice benchmarks.

Each question becomes what the model was trained on: a situation (text, with the passage first when there is one,
then "Question: ...") and a list of options. The model returns one score per option and we count how often the
right option scores highest.

Benchmarks (all downloaded with the Hugging Face `datasets` library, so this needs internet once):
    race            passage + question + 4 options. Closest to the training data, so the most meaningful one.
    boolq           passage + yes/no question. Beat the "most common position" row, not 50%.
    arc-easy        science question, no passage, about 4 options.
    arc-challenge   harder science question, no passage. Expect near chance.
    csqa            CommonsenseQA, 5 short options, no passage.

For every benchmark you get a table with:
    chance                      guessing
    longest option              always pick the longest option (a cheap trick that sometimes works)
    most common position        always pick option A / B / ... whichever is right most often
    raw Qwen vectors            the frozen backbone with NO trained heads (the number the heads must beat)
    after 'pre' / 'mid' / 'post'  the trained model, stage by stage
    Qwen log-likelihood         the usual way to score a small LM: the SAME Qwen, one forward pass per option, pick the option
                                with the highest log-probability (acc_norm divides by the option's length). This is the
                                yardstick for the CLM's speed claim: how much accuracy do you give up for it?
    bge-small / all-MiniLM      ready-made sentence embedders used as they are, no training (see baselines.py).
                                If one of them beats the trained heads, say so in the README.
All rows use the SAME sampled questions and the same text, so they can be compared directly.
The +/- is a 95% margin of error: two numbers closer than that are not clearly different.

How to read it: this is a small frozen scorer, not a language model. It compares the meaning of the situation with the
meaning of each option and does no step-by-step reasoning. Do not put its numbers next to LLM leaderboards.
Use the table to see (a) whether the trained heads beat raw Qwen on data they never saw and (b) where it stays near chance.

Run:
    python benchmark.py                          # all benchmarks, 1000 random questions each
    python benchmark.py --bench race             # one benchmark
    python benchmark.py --bench race --limit 0   # every question (slower)
    python benchmark.py --stage post             # only one stage of the trained heads
    python benchmark.py --baselines none         # skip the baseline rows (much faster)
    python benchmark.py --baselines ll           # only the Qwen log-likelihood rows  (or: embedders / all)
"""
import argparse
import math
import random

import numpy as np
import torch

import baselines
import clm


# ===================================== DATA =====================================
def load_split(names, config, split):
    """Download a dataset from the Hugging Face hub. Tries each name in turn (names sometimes move)."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("This script needs the Hugging Face `datasets` library. Install it with: pip install datasets")
    last_error = None
    for name in names:
        try:
            return load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
        except Exception as error:               # wrong name, no internet, ... try the next name
            last_error = error
    raise SystemExit(f"Could not download {names[0]} ({split}): {last_error}\n"
                     "Check your internet connection, or the dataset name on huggingface.co.")


def build_race():
    rows = load_split(["ehovy/race"], "all", "test")
    return [{"state": f"{r['article'].strip()}\n\nQuestion: {r['question'].strip()}",
             "options": [str(o) for o in r["options"]], "answer": "ABCD".index(r["answer"])} for r in rows]


def build_boolq():
    rows = load_split(["google/boolq", "boolq"], None, "validation")   # the test split has no labels
    return [{"state": f"{r['passage'].strip()}\n\nQuestion: {r['question'].strip()}?",
             "options": ["Yes", "No"], "answer": 0 if r["answer"] else 1} for r in rows]


def build_arc(config):
    rows = load_split(["allenai/ai2_arc", "ai2_arc"], config, "test")
    cases = []
    for r in rows:
        labels, texts = list(r["choices"]["label"]), list(r["choices"]["text"])
        if r["answerKey"] in labels:
            cases.append({"state": f"Question: {r['question'].strip()}", "options": [str(t) for t in texts],
                          "answer": labels.index(r["answerKey"])})
    return cases


def build_csqa():
    rows = load_split(["tau/commonsense_qa", "commonsense_qa"], None, "validation")   # the test split has no labels
    cases = []
    for r in rows:
        labels, texts = list(r["choices"]["label"]), list(r["choices"]["text"])
        if r["answerKey"] in labels:
            cases.append({"state": f"Question: {r['question'].strip()}", "options": [str(t) for t in texts],
                          "answer": labels.index(r["answerKey"])})
    return cases


BENCHMARKS = {
    "race": build_race,
    "boolq": build_boolq,
    "arc-easy": lambda: build_arc("ARC-Easy"),
    "arc-challenge": lambda: build_arc("ARC-Challenge"),
    "csqa": build_csqa,
}


# ==================================== SCORING ====================================
@torch.inference_mode()
def predict(scorer, cases):
    """Return {system name: [index of the option it picks, per case]} for raw Qwen vectors and every trained stage."""
    # Each distinct option text is embedded once (yes/no and repeated answers are common).
    texts, text_index, option_rows = [], {}, []
    for c in cases:
        rows = []
        for option in c["options"]:
            if option not in text_index:
                text_index[option] = len(texts)
                texts.append(option)
            rows.append(text_index[option])
        option_rows.append(rows)

    print(f"  embedding {len(cases)} situations and {len(texts)} distinct options ...")
    S = scorer.state_vectors([c["state"] for c in cases])                      # the SAME recipe as training
    A = scorer.action_vectors(texts)

    systems = {"raw Qwen vectors": (None, None)}                                # no heads = the frozen backbone alone
    for stage, (state_head, action_head, _) in scorer.heads.items():
        systems[f"after '{stage}'"] = (state_head, action_head)

    predictions = {}
    for name, (state_head, action_head) in systems.items():
        zs = S if state_head is None else state_head(S)
        za = A if action_head is None else action_head(A)
        predictions[name] = [int((za[rows] @ zs[i]).argmax()) for i, rows in enumerate(option_rows)]
    return predictions


def accuracy_text(correct, n):
    p = correct / n
    margin = 1.96 * math.sqrt(p * (1 - p) / n)
    return f"{p:6.1%} +/- {margin:4.1%}"


def run_benchmark(name, scorer, limit, seed, baseline_names):
    print(f"\n=== {name} ===")
    cases = BENCHMARKS[name]()
    if limit and limit < len(cases):
        cases = random.Random(seed).sample(cases, limit)
    n = len(cases)
    print(f"  {n} questions")

    answers = [c["answer"] for c in cases]
    rows = [("chance (guessing)", sum(1 / len(c["options"]) for c in cases))]
    longest = [max(range(len(c["options"])), key=lambda k: len(c["options"][k])) for c in cases]
    rows.append(("longest option", sum(p == a for p, a in zip(longest, answers))))
    common = max(set(answers), key=answers.count)
    rows.append(("most common position", sum(a == common for a in answers)))
    all_preds = predict(scorer, cases)
    all_preds.update(baselines.predict(cases, scorer.tok, baseline_names))   # the SAME questions for every row
    for system, preds in all_preds.items():
        rows.append((system, sum(p == a for p, a in zip(preds, answers))))

    print(f"  {'':32s} {'accuracy (95% margin)':>24s}")
    for label, correct in rows:
        print(f"  {label:32s} {accuracy_text(correct, n):>24s}")
    return n, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", choices=[*BENCHMARKS, "all"], default="all")
    parser.add_argument("--limit", type=int, default=1000, help="random questions per benchmark (0 = all of them)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage", choices=["pre", "mid", "post", "all"], default="all",
                        help="which trained heads to test (default: all that exist)")
    parser.add_argument("--baselines", choices=["none", "ll", "embedders", "all"], default="all",
                        help="extra rows to compare against: Qwen log-likelihood (ll), ready-made embedders, both (all), or none")
    args = parser.parse_args()

    scorer = clm.Scorer(("pre", "mid", "post") if args.stage == "all" else (args.stage,))
    names = list(BENCHMARKS) if args.bench == "all" else [args.bench]
    baseline_names = baselines.parse(args.baselines)
    results = {name: run_benchmark(name, scorer, args.limit, args.seed, baseline_names) for name in names}

    if len(names) > 1:                                     # one summary table with the systems side by side
        labels = [label for label, _ in results[names[0]][1]]
        print("\n================ SUMMARY (accuracy) ================")
        print(f"{'':32s}" + "".join(f"{name:>15s}" for name in names))
        for label in labels:
            cells = ""
            for name in names:
                n, rows = results[name]
                correct = dict(rows)[label]
                cells += f"{correct / n:15.1%}"
            print(f"{label:32s}{cells}")
    print("\nRead it like this: a trained row clearly above 'raw Qwen vectors' means the heads helped on data they never saw. "
          "A row near 'chance' means the model cannot do that kind of question.\n"
          "Compare with 'Qwen log-likelihood' to see what accuracy the CLM gives up for its speed (bench_latency.py has the speed).\n"
          "The (acc_norm) row is not meaningful for BoolQ (Yes and No differ in length); read the (acc) row there.")


if __name__ == "__main__":
    main()
