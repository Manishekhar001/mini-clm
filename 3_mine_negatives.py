"""Step 3 - find "hard negatives" for the Q&A pairs (used by stage 2 and by the replay in stage 3).

A hard negative is a WRONG answer that looks plausible. Mid-training shows the model each question
together with up to 10 of them, which forces it to learn finer differences than "random answer".

How we find them: for each pair, take REAL answers from OTHER documents whose vectors look most
like the correct answer. No LLM and no extra embedding needed.

Two kinds of answers are skipped, because they would be unfairly "wrong":
  - answers from the same document (they can be correct for a similar question)
  - answers whose text is identical to the correct one (like "Yes." or "No.")

Training pairs only get negatives from the training pairs, and test pairs only from the test pairs.

If 2b_write_negatives.py was run, the wrong answers it WROTE for a question take the first slots of that
question (up to 3), and the mined ones fill the remaining slots.

Run:
    python 3_mine_negatives.py
"""
import json
import os
import zlib

import numpy as np
import torch

import clm

NEG_PER_PAIR = 10    # up to 10 hard negatives per pair
NEG_WRITTEN = 3      # at most this many of them are written ones (from 2b_write_negatives.py)


def ascii_safe(text, n=90):
    """Shorten and strip non-ASCII characters so printing never fails on a Windows console."""
    return " ".join(text.split())[:n].encode("ascii", "replace").decode()


def main():
    device = clm.get_device()
    data = np.load(clm.emb_path("qa"))
    if str(data["model_name"]) != clm.MODEL_NAME:
        raise SystemExit(f"Q&A vectors were made with {data['model_name']} but clm.py says {clm.MODEL_NAME}. Rerun step 2.")
    with open(clm.QA_PAIRS_PATH, encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f]
    n = len(pairs)
    if n != len(data["actions"]):
        raise SystemExit("qa_pairs.jsonl and the Q&A vectors do not match. Rerun step 2.")

    actions = torch.from_numpy(data["actions"]).to(device).float()
    group = torch.from_numpy(data["group"]).to(device)
    # A number that is the same for answers with the same text (ignoring case and spacing).
    text_key = torch.tensor([zlib.crc32(" ".join(p["action"].lower().split()).encode()) for p in pairs],
                            dtype=torch.int64, device=device)
    is_val = data["is_val"]

    neg_idx = np.full((n, NEG_PER_PAIR), -1, dtype=np.int64)       # -1 = "no negative here"
    for pool in (np.flatnonzero(~is_val), np.flatnonzero(is_val)):
        pool_t = torch.from_numpy(pool).to(device)
        pool_actions, pool_group, pool_key = actions[pool_t], group[pool_t], text_key[pool_t]
        k = min(NEG_PER_PAIR, len(pool))
        for start in range(0, len(pool), 1024):
            rows = slice(start, start + 1024)
            sims = pool_actions[rows] @ pool_actions.T                       # similarity of every answer to the correct one
            bad = (pool_group[rows, None] == pool_group[None, :]) | (pool_key[rows, None] == pool_key[None, :])
            sims = sims.masked_fill(bad, -2.0)                               # never pick same-document / same-text answers
            values, top = sims.topk(k, dim=1)                                # the most similar remaining answers
            top = pool_t[top].masked_fill(values <= -2.0, -1)                # back to global indices; -1 if nothing valid
            neg_idx[pool[start:start + 1024], :k] = top.cpu().numpy()

    # Add the negatives WRITTEN by 2b_write_negatives.py (optional). Their vectors are stored after the pairs, so a
    # number >= n in neg_idx means "written negative number (number - n)".
    extra, extra_texts = np.zeros((0, data["actions"].shape[1]), dtype=np.float16), []
    if os.path.exists(clm.SYN_NEG_PATH):
        syn = np.load(clm.SYN_NEG_PATH)
        if str(syn["model_name"]) != clm.MODEL_NAME:
            print(f"(written negatives skipped: made with {syn['model_name']} but clm.py says {clm.MODEL_NAME}. Rerun 2b.)")
        else:
            keep, by_pair = [], {}
            for j, (p, k) in enumerate(zip(syn["pair_idx"], syn["keys"])):
                if p < n and clm.pair_key(pairs[p]) == k:    # still belongs to the same question and answer
                    by_pair.setdefault(int(p), []).append(len(keep))
                    keep.append(j)
            extra = syn["vectors"][keep]
            extra_texts = [str(t) for t in syn["texts"][keep]]
            for i, rows in by_pair.items():                  # written first, then the mined ones, 10 in total
                mined = [int(x) for x in neg_idx[i] if x >= 0]
                merged = ([n + r for r in rows[:NEG_WRITTEN]] + mined)[:NEG_PER_PAIR]
                neg_idx[i] = -1
                neg_idx[i, :len(merged)] = merged
            print(f"Added {len(keep)} written negatives for {len(by_pair)} pairs ({len(syn['pair_idx']) - len(keep)} stale ones ignored).")
    np.savez(clm.NEG_PATH, neg_idx=neg_idx, extra=extra)
    counts = (neg_idx >= 0).sum(axis=1)
    print(f"Negatives for {n} pairs; on average {counts.mean():.1f} per pair ({(counts == 0).sum()} pairs have none).")

    # Look at a few by eye - this is the best way to see if the negatives make sense.
    shown = list(np.random.default_rng(0).choice(n, 3, replace=False))
    with_written = [i for i in range(n) if (neg_idx[i] >= n).any()]
    if with_written:                                         # also show a few pairs that have written negatives
        shown += list(np.random.default_rng(1).choice(with_written, min(3, len(with_written)), replace=False))
    for i in shown:
        print("\n  Question:       ", ascii_safe(clm.question_of(pairs[i])))
        print("  Correct answer: ", ascii_safe(pairs[i]["action"]))
        for j in neg_idx[i][:3]:
            if j >= n:
                print("  Wrong (written):", ascii_safe(extra_texts[j - n]))
            elif j >= 0:
                print("  Wrong (mined):  ", ascii_safe(pairs[j]["action"]))
    print("\nNext: python 4_train.py --stage all")


if __name__ == "__main__":
    main()
