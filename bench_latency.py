"""Latency (optional) - how long does ONE question take? The speed claim of a CLM is exactly this table.

Every question is answered at batch size 1 (one question at a time, like a real user), and we time:

    CLM (full)             situation -> Qwen -> state head, options -> Qwen -> action head, cosine, pick. Everything.
    CLM (options cached)   the same, but the option vectors were computed BEFORE the timer starts. This is the real
                           situation when the options are known in advance (the agent's possible actions, a fixed
                           list of labels). It does NOT apply when every question brings its own new options.
    Qwen log-likelihood (4 passes)         the plain method: one Qwen forward pass per option, each reading the whole situation.
    Qwen log-likelihood (1 batch of 4)     the same four sequences in a single batch (a fairer, faster way to do it).

How the timing is done:
  * `torch.cuda.synchronize()` before and after each question, because the GPU runs in the background and without it
    the clock would stop before the work is finished.
  * 10 warm-up questions per method first (not timed): the first runs are slow because the GPU is still starting up.
  * The methods are timed in turns on the SAME question (the order rotates), so a laptop GPU that heats up and slows
    down hurts all methods equally. Plug the laptop in and use the performance power mode.
  * Tokenizing and the small CPU steps are INSIDE the timer (a user would wait for them too).

Run:
    python bench_latency.py                        # RACE (long passages) and ARC-Easy (short questions), 200 questions each
    python bench_latency.py --bench race --n 200
    python bench_latency.py --stage mid            # time the heads of another stage (default: post)
"""
import argparse
import random
import time

import numpy as np
import torch

import baselines
import benchmark
import clm


def pick_cases(bench, n, seed):
    """n random questions with exactly 4 options (the same builders benchmark.py uses)."""
    cases = [c for c in benchmark.BENCHMARKS[bench]() if len(c["options"]) == 4]
    return random.Random(seed).sample(cases, n)


def make_methods(scorer, stage, lm, cases):
    """The things to time. Each one takes the index of a question and returns the index of the option it picks."""
    tok = scorer.tok
    state_head, action_head, _ = scorer.heads[stage]

    with torch.inference_mode():                         # option vectors for the "cached" method, made BEFORE any timing
        cached = [action_head(scorer.action_vectors(c["options"])) for c in cases]

    def clm_full(i):
        probs = scorer.rank(cases[i]["state"], cases[i]["options"])[stage]
        return int(np.argmax(probs))

    @torch.inference_mode()
    def clm_cached(i):
        state_vec = scorer.state_vectors([cases[i]["state"]])
        return int((state_head(state_vec) @ cached[i].T).argmax())

    def qwen_ll(i, batched):
        ctx_ids, cont_ids = baselines.encode_case(tok, cases[i]["state"], cases[i]["options"])
        return int(np.argmax(baselines.option_loglik(lm, ctx_ids, cont_ids, tok.pad_token_id, batched=batched)))

    return {"CLM (full)": clm_full,
            "CLM (options cached)": clm_cached,
            "Qwen log-likelihood (4 passes)": lambda i: qwen_ll(i, False),
            "Qwen log-likelihood (1 batch of 4)": lambda i: qwen_ll(i, True)}


def run(bench, scorer, stage, lm, n, warmup, seed):
    cases = pick_cases(bench, n + warmup, seed)
    methods = make_methods(scorer, stage, lm, cases)
    names = list(methods)

    print(f"\n[{bench}] warming up ({warmup} questions per method) ...")
    for name in names:
        for i in range(warmup):
            methods[name](i)

    print(f"[{bench}] timing {n} questions x {len(names)} methods ...")
    times_ms = {name: [] for name in names}
    picks = {name: [] for name in names}
    for k, i in enumerate(range(warmup, warmup + n)):
        turn = k % len(names)
        for name in names[turn:] + names[:turn]:         # the order rotates from question to question
            torch.cuda.synchronize()
            start = time.perf_counter()
            pick = methods[name](i)
            torch.cuda.synchronize()
            times_ms[name].append((time.perf_counter() - start) * 1000)
            picks[name].append(pick)

    lengths = [len(clm.state_to_ids(scorer.tok, [{"role": "user", "content": c["state"]}])) for c in cases[warmup:]]
    slowest = "Qwen log-likelihood (4 passes)"
    base = np.median(times_ms[slowest])
    print(f"\n=== {bench}: {n} questions, 4 options each, batch size 1, median situation {np.median(lengths):.0f} tokens ===")
    print("| method | median (ms) | p95 (ms) | speed-up vs 4-pass Qwen (median) |")
    print("|---|---|---|---|")
    for name in names:
        t = np.array(times_ms[name])
        print(f"| {name} | {np.median(t):.1f} | {np.percentile(t, 95):.1f} | {base / np.median(t):.1f}x |")

    same = lambda a, b: np.mean(np.array(picks[a]) == np.array(picks[b]))
    print(f"sanity: CLM full vs cached pick the same option on {same('CLM (full)', 'CLM (options cached)'):.1%} of questions; "
          f"Qwen 4 passes vs 1 batch on {same(slowest, 'Qwen log-likelihood (1 batch of 4)'):.1%} (both should be about 100%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", nargs="+", choices=["race", "arc-easy", "arc-challenge", "csqa"], default=["race", "arc-easy"])
    parser.add_argument("--n", type=int, default=200, help="questions timed per benchmark")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--stage", choices=["pre", "mid", "post"], default="post", help="which trained heads to time")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    scorer = clm.Scorer((args.stage,))
    lm = baselines.get_lm()
    print(f"dtype {next(lm.parameters()).dtype}; GPU memory in use after loading both models: "
          f"{torch.cuda.memory_allocated() / 1024 ** 3:.1f} GB")
    for bench in args.bench:
        run(bench, scorer, args.stage, lm, args.n, args.warmup, args.seed)
    print("\nPaste the tables into the README. Mention the GPU, the dtype and that questions run one at a time (batch size 1).")
    print("'CLM (options cached)' only applies when the options are known in advance; say so next to the number.")


if __name__ == "__main__":
    main()
