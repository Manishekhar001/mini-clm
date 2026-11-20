"""Step 2 - turn every state and every action into a vector with the frozen Qwen backbone.

Does it for both datasets: the Q&A pairs (stages 1 and 2) and the agent pairs (stage 3).

How a text becomes a vector is set in clm.py (the "recipe": plain text, mean pooling, centering).
After embedding, the average state vector and average action vector of the Q&A TRAINING pairs are
subtracted from every vector (centering). They are saved in data/centering.npz so that the demo and
2b_write_negatives.py can do exactly the same. That is why the Q&A dataset is embedded first.

This is the slow step (the GPU reads every text once). It saves its progress every 1000 pairs,
so if you stop it (Ctrl+C) or the laptop sleeps, just run it again and it carries on.

Run:
    python 2_embed.py                # both datasets
    python 2_embed.py --only qa      # just one of them ("qa" or "agent"; do "qa" first)
"""
import argparse
import glob
import json
import os
import shutil
import time
import zlib

import numpy as np

import clm

CHUNK = 1000   # pairs per saved chunk


def embed_file(kind, pairs_path, tok, model):
    with open(pairs_path, encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f]
    n = len(pairs)
    print(f"\n[{kind}] {n} pairs to embed with {clm.MODEL_NAME}")
    print(f"[{kind}] recipe: {clm.RECIPE}  (id {clm.RECIPE_ID})")

    # Chunks live in a folder named after the data kind, model, recipe and size, so vectors from a different setup never get mixed in.
    chunk_dir = os.path.join(clm.DATA_DIR, f"chunks_{kind}_{clm.MODEL_NAME.replace('/', '_')}_{clm.RECIPE_ID}_{n}")
    os.makedirs(chunk_dir, exist_ok=True)
    for leftover in glob.glob(os.path.join(chunk_dir, "*.tmp.npz")):   # a half-written file from an interrupted run
        os.remove(leftover)
    t0, done_now = time.time(), 0

    for c, start in enumerate(range(0, n, CHUNK)):
        path = os.path.join(chunk_dir, f"{c:05d}.npz")
        if os.path.exists(path):             # finished in an earlier run
            continue
        part = pairs[start:start + CHUNK]
        state_ids = [clm.state_to_ids(tok, p["state"]) for p in part]
        action_ids = [clm.action_to_ids(tok, p["action"]) for p in part]
        states = clm.embed(model, tok, state_ids)
        actions = clm.embed(model, tok, action_ids)
        np.savez(path + ".tmp.npz", states=states, actions=actions)
        os.replace(path + ".tmp.npz", path)  # only a fully written chunk counts as done

        done_now += len(part)
        rate = done_now / (time.time() - t0)
        left_min = (n - start - len(part)) / rate / 60
        print(f"  [{kind}] {start + len(part)}/{n} pairs  ({rate:.1f} pairs/s, about {left_min:.0f} min left)")

    # Put all chunks together into one file.
    files = sorted(glob.glob(os.path.join(chunk_dir, "[0-9][0-9][0-9][0-9][0-9].npz")))   # finished chunks only
    states = np.concatenate([np.load(f)["states"] for f in files])
    actions = np.concatenate([np.load(f)["actions"] for f in files])
    # "group" says which document / recording a pair came from; pairs of one group are never used as each other's wrong answers.
    group = np.array([zlib.crc32((p.get("doc") or p["traj"]).encode()) for p in pairs], dtype=np.int64)
    is_val = np.array([p["split"] == "val" for p in pairs])

    # Centering: remove the part of every vector that all texts share.
    print(f"[{kind}] mean cosine between different texts, before centering: "
          f"states {clm.mean_pairwise_cosine(states):.3f}, actions {clm.mean_pairwise_cosine(actions):.3f}")
    if clm.CENTER:
        if kind == "qa":                                       # the averages come from Q&A TRAINING pairs only
            state_mean = states[~is_val].astype(np.float32).mean(axis=0)
            action_mean = actions[~is_val].astype(np.float32).mean(axis=0)
            clm.save_centering(state_mean, action_mean)
            print(f"[{kind}] saved the average vectors to {clm.CENTER_PATH}")
        else:                                                  # agent pairs use the SAME averages (the heads are shared)
            state_mean, action_mean = clm.load_centering()
        states = clm.center_vectors(states, state_mean)
        actions = clm.center_vectors(actions, action_mean)
        print(f"[{kind}] after centering: states {clm.mean_pairwise_cosine(states):.3f}, "
              f"actions {clm.mean_pairwise_cosine(actions):.3f}")
    state_cos = clm.mean_pairwise_cosine(states)
    if state_cos > 0.9:
        print(f"[{kind}] WARNING: the state vectors are still almost identical to each other (cosine {state_cos:.2f}). "
              "Training cannot tell questions apart from vectors like that. Check the recipe in clm.py.")

    np.savez(clm.emb_path(kind), states=states, actions=actions, group=group, is_val=is_val,
             model_name=np.array(clm.MODEL_NAME), recipe=np.array(clm.RECIPE_ID))
    shutil.rmtree(chunk_dir)
    print(f"[{kind}] saved {len(states)} state vectors and action vectors to {clm.emb_path(kind)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["qa", "agent"], help="embed just one dataset")
    args = parser.parse_args()

    jobs = [("qa", clm.QA_PAIRS_PATH, "1a_build_qa_pairs.py"),
            ("agent", clm.AGENT_PAIRS_PATH, "1b_build_agent_pairs.py")]
    jobs = [j for j in jobs if args.only in (None, j[0])]
    for kind, path, script in jobs:
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found. Run {script} first.")

    tok, model = clm.load_backbone()
    for kind, path, _ in jobs:
        embed_file(kind, path, tok, model)
    print("\nNext: python 2b_write_negatives.py (optional), then python 3_mine_negatives.py")


if __name__ == "__main__":
    main()
