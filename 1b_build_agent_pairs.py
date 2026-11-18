"""Step 1b - build the AGENT pairs, used in the third (post-training) stage.

A "pair" is (state, action):
    state  = what the agent could see at that moment (the task and the conversation so far)
    action = what the agent decided to do next (its message and/or its tool calls)

Source: Agent Data Protocol v2 (neulab/adp-v2), a public collection of recordings of AI agents
solving tasks. It is plain text, so we can embed it with any backbone.

It runs on your laptop and only needs `requests` and `huggingface_hub` - no GPU, no torch.

Run:
    python 1b_build_agent_pairs.py                    # about 300 pairs from each dataset config
    python 1b_build_agent_pairs.py --per-config 100   # smaller and faster
"""
import argparse
import glob
import json
import os
import random
import shutil
import zlib

import requests
from huggingface_hub import HfApi

REPO = "neulab/adp-v2"
NEEDS_IMAGES = {"android_in_the_wild", "androidcontrol", "omniact", "screenagent", "llava_plus"}  # text-only model: skip
PAIRS_PER_RECORDING = 3     # at most this many pairs from one recording (more recordings = more variety)
MAX_MSG_CHARS = 8000        # cut every single message to this many characters
STATE_CHARS = 6000          # a state keeps only its most recent ~6000 characters (we cut to 512 tokens later anyway)
VAL_PERCENT = 10            # about 10% of RECORDINGS are held out for testing (never trained on)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")   # same folder as clm.py uses
PAIRS_PATH = os.path.join(DATA_DIR, "agent_pairs.jsonl")


def obs_text(observation):
    """The text of what a tool returned (terminal output, web page, ...)."""
    if not isinstance(observation, dict):
        return ""
    parts = []
    for result in observation.get("results") or []:
        content = result.get("content")
        if not content:
            content = ((result.get("extra") or {}).get("web") or {}).get("axtree")
        if content:
            parts.append(content if isinstance(content, str) else json.dumps(content))
    return "\n".join(parts)


def action_text(step):
    """What the agent did: its message, plus each tool call written as name(json arguments)."""
    parts = []
    if isinstance(step.get("message"), str) and step["message"].strip():
        parts.append(step["message"])
    for call in step.get("tool_calls") or []:
        parts.append(f"{call.get('function_name')}({json.dumps(call.get('arguments'), ensure_ascii=False)})")
    return "\n".join(parts).strip()


def trim_history(history):
    """Keep only the newest messages, about STATE_CHARS characters in total."""
    kept, chars = [], 0
    for msg in reversed(history):
        kept.append(msg)
        chars += len(msg["content"])
        if chars >= STATE_CHARS:
            break
    return kept[::-1]


def pairs_from(record, rng):
    """One recording -> a few (state messages, action text, step number) tuples."""
    steps = record.get("steps") or []
    acting = [s for s in steps if s.get("source") == "agent" and action_text(s)]
    if len(acting) < 2:                      # skip recordings where the agent barely does anything
        return []

    history, pairs = [], []
    for i, step in enumerate(steps):
        source, message = step.get("source"), step.get("message")
        if source in ("system", "user"):
            if isinstance(message, str) and message:
                history.append({"role": source, "content": message[:MAX_MSG_CHARS]})
        elif source == "agent":
            action = action_text(step)
            if action:
                if history:                  # state = everything so far, action = this decision
                    pairs.append((trim_history(history), action[:MAX_MSG_CHARS], i))
                history.append({"role": "assistant", "content": action[:MAX_MSG_CHARS]})
            observation = obs_text(step.get("observation"))
            if observation:                  # tool output becomes part of the next state
                history.append({"role": "user", "content": observation[:MAX_MSG_CHARS]})
    rng.shuffle(pairs)
    return pairs[:PAIRS_PER_RECORDING]


def build_config(api, config, quota):
    """Stream recordings of one dataset config until we have `quota` pairs. Returns a list of rows."""
    rng = random.Random(zlib.crc32(config.encode()))
    files = sorted(x.path for x in api.list_repo_tree(REPO, f"{config}/raw", repo_type="dataset")
                   if x.path.endswith(".jsonl") and "full_std" in x.path)
    rows, n_records = [], 0
    for path in files:
        if len(rows) >= quota:
            break
        url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{path}"
        with requests.get(url, stream=True, timeout=120) as resp:     # stream: we stop early, no huge download
            resp.raise_for_status()
            for line in resp.iter_lines():
                if len(rows) >= quota:
                    break
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                n_records += 1
                traj_id = f"{config}:{record.get('trajectory_id') or record.get('session_id') or n_records}"
                # The split is decided per RECORDING (by hashing its id), so the test set never shares a recording with training.
                split = "val" if zlib.crc32(traj_id.encode()) % 100 < VAL_PERCENT else "train"
                for state, action, step in pairs_from(record, rng):
                    rows.append({"config": config, "traj": traj_id, "step": step, "split": split,
                                 "state": state, "action": action})
    return rows[:quota]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-config", type=int, default=300, help="max pairs taken from each dataset config")
    args = parser.parse_args()

    api = HfApi()
    configs = [x.path for x in api.list_repo_tree(REPO, repo_type="dataset")
               if not x.path.startswith(".") and "." not in x.path]      # folders only
    configs = [c for c in configs if c not in NEEDS_IMAGES]
    print(f"{len(configs)} dataset configs; taking up to {args.per_config} pairs from each")

    all_rows = []
    for config in configs:
        try:
            rows = build_config(api, config, args.per_config)
        except Exception as e:               # one broken config should not stop the whole run
            print(f"  {config}: skipped ({type(e).__name__}: {str(e)[:120]})")
            continue
        print(f"  {config}: {len(rows)} pairs")
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("No pairs were built. Check your internet connection and try again.")

    random.Random(0).shuffle(all_rows)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(PAIRS_PATH, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # New pairs make any old agent vectors invalid, so remove them.
    for old in glob.glob(os.path.join(DATA_DIR, "chunks_agent_*")):
        shutil.rmtree(old)
    old_vectors = os.path.join(DATA_DIR, "embeddings_agent.npz")
    if os.path.exists(old_vectors):
        os.remove(old_vectors)

    n_val = sum(r["split"] == "val" for r in all_rows)
    print(f"\nSaved {len(all_rows)} pairs ({len(all_rows) - n_val} train, {n_val} test) to {PAIRS_PATH}")
    print("Next: python 2_embed.py (after 1a_build_qa_pairs.py too)")


if __name__ == "__main__":
    main()
