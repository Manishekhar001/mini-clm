"""Stage-3 evaluation (optional) - did the agent stage ('post') learn anything that stage 2 ('mid') did not already know?

The task: given what the agent could see (the task and the history so far = the "state"), rank the action it really took
among a list of candidate actions.

The test steps are the held-out recordings only. 1b_build_agent_pairs.py already splits per RECORDING (a hash of the
recording id, about 10% held out), so no test step shares a recording with a training step. This script checks that
again and does not need a retrain.

Two candidate sets (the SAME candidates are used for every system, so the comparison is fair):
    same-config   the true action + up to 15 actions from OTHER held-out recordings of the SAME dataset config (all
                  terminal agents, all web agents, ...). Plausible wrong actions: this is the test to trust.
    any-config    the true action + 15 held-out actions from any other recording. This is what 4_train.py prints as
                  'Agent (1 of 16)'. It is easy ("a terminal command vs a web click") and can hide a weak model, the
                  same trap as lesson 2 in the README.
Candidates with the same text as the true action are dropped (they would not be wrong). A different random draw of
candidates is used each round (--draws), and the rounds are averaged.

Systems:
    random guessing   exact value, 1 / number of candidates
    prior             the most common action TYPE (the tool name, or "(message)") of the training data of that config.
                      Actions are almost all unique text, so "the most frequent action" has to be read as a type.
                      It knows the dataset and nothing else. A model that does not beat it only learned "what usually happens".
    raw Qwen vectors  the frozen backbone with no heads
    after pre / mid / post   the trained heads, stage by stage

Numbers are top-1 and top-3. The 95% interval resamples whole RECORDINGS, because the steps of one recording are not
independent. The last block compares two systems on exactly the same steps; 'post' against 'mid' is the only number that
shows whether stage 3 learned anything.

Run:
    python eval_stage3.py
    python eval_stage3.py --draws 10 --k 16
"""
import argparse
import json
import os
import re
import zlib
from collections import Counter, defaultdict

import numpy as np
import torch

import clm

KS = (1, 3)                 # top-1 and top-3
MIN_DISTRACTORS = 3         # a step needs at least this many wrong candidates to be tested
PRIOR = "prior (common action type)"
RANDOM = "random guessing"

# An action is written by 1b as optional text lines and then one line per tool call: name({json arguments})
TOOL_CALL = re.compile(r"^([A-Za-z_][\w.\-:]*)\((?:\{|null|\[|\"|\d|true|false)")


def action_type(text):
    """The name of the first tool call in an action, or "(message)" if the agent only wrote text."""
    for line in text.split("\n"):
        match = TOOL_CALL.match(line)
        if match:
            return match.group(1)
    return "(message)"


# ======================================= DATA =======================================
def load_agent_data():
    """The agent vectors (made by 2_embed.py) together with the text of each pair (made by 1b), checked to be the same rows."""
    path = clm.emb_path("agent")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Run 1b_build_agent_pairs.py and 2_embed.py first.")
    d = np.load(path)
    if "recipe" not in d.files or str(d["recipe"]) != clm.RECIPE_ID:
        raise SystemExit("The agent vectors were made with a different embedding recipe than clm.py says now. Rerun step 2.")
    states, actions, saved_group, saved_val = d["states"], d["actions"], d["group"], d["is_val"]

    with open(clm.AGENT_PAIRS_PATH, encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f]
    group = np.array([zlib.crc32((p.get("doc") or p["traj"]).encode()) for p in pairs], dtype=np.int64)
    is_val = np.array([p["split"] == "val" for p in pairs])
    if len(pairs) != len(states) or not (group == saved_group).all() or not (is_val == saved_val).all():
        raise SystemExit("agent_pairs.jsonl and embeddings_agent.npz are not the same rows (one was rebuilt without the other).\n"
                         "Run `python 2_embed.py --only agent` again.")
    texts = [p["action"] for p in pairs]
    return {"S": states, "A": actions, "group": group, "is_val": is_val,
            "config": np.array([p["config"] for p in pairs]),
            "type": np.array([action_type(t) for t in texts]),
            "text_id": np.array([zlib.crc32(t.encode()) for t in texts], dtype=np.int64)}


def train_prior(D):
    """How often each action type occurs in the TRAINING pairs, per config (and overall, for configs never seen in training)."""
    per_config, overall = defaultdict(Counter), Counter()
    train = ~D["is_val"]
    for config, kind in zip(D["config"][train], D["type"][train]):
        per_config[config][kind] += 1
        overall[kind] += 1
    return per_config, overall


# ================================= CANDIDATES AND SCORES =================================
def make_candidates(V, same_config, k, rng):
    """Row r is held-out step r. Column 0 = its true action, the other columns = wrong candidates (-1 where there are fewer).
    All numbers are positions in the held-out arrays V."""
    n = len(V["group"])
    cand = np.full((n, k), -1, dtype=np.int64)
    cand[:, 0] = np.arange(n)
    by_config = defaultdict(list)
    for j in range(n):
        by_config[V["config"][j]].append(j)
    by_config = {c: np.array(v) for c, v in by_config.items()}
    everything = np.arange(n)
    for r in range(n):
        pool = by_config[V["config"][r]] if same_config else everything
        ok = pool[(V["group"][pool] != V["group"][r]) & (V["text_id"][pool] != V["text_id"][r])]   # other recording, other text
        if len(ok) < MIN_DISTRACTORS:
            continue                                   # too few wrong candidates: this step is skipped (it stays all -1)
        take = rng.choice(ok, size=min(k - 1, len(ok)), replace=False)
        cand[r, 1:1 + len(take)] = take
    return cand


def candidate_scores(zs, za, cand, chunk=512):
    """cosine(state r, candidate action) for every candidate; -inf where there is no candidate."""
    out = np.full(cand.shape, -np.inf, dtype=np.float32)
    for start in range(0, len(cand), chunk):
        c = cand[start:start + chunk]
        sim = np.einsum("nd,nkd->nk", zs[start:start + chunk], za[np.maximum(c, 0)])
        out[start:start + chunk] = np.where(c >= 0, sim, -np.inf)
    return out


def top_hits(scores):
    """{k: 1.0 if the true action (column 0) is among the k best candidates, else 0.0}, for every step."""
    better = (scores[:, 1:] > scores[:, :1]).sum(axis=1)
    return {k: (better < k).astype(float) for k in KS}


def expected_top_hits(scores):
    """Like top_hits, but a tie is broken at random (the exact expected value). Needed for the prior, which has many ties."""
    better = (scores[:, 1:] > scores[:, :1]).sum(axis=1)
    tied = (scores[:, 1:] == scores[:, :1]).sum(axis=1)
    return {k: np.clip((k - better) / (tied + 1), 0.0, 1.0) for k in KS}


def prior_scores(V, cand, prior):
    """Score of a candidate = how common its action type is in the training data of the config the STATE comes from."""
    per_config, overall = prior
    scores = np.full(cand.shape, -np.inf)
    for r in range(len(cand)):
        table = per_config.get(V["config"][r]) or overall
        total = sum(table.values()) or 1
        for c, j in enumerate(cand[r]):
            if j >= 0:
                scores[r, c] = table.get(V["type"][j], 0) / total
    return scores


def evaluate_set(V, vectors, prior, same_config, args):
    """Hit rates (0..1 per step, averaged over the random draws of candidates) for every system.

    vectors: {system name: (state vectors, action vectors)} of the held-out steps, as unit-length numpy arrays.
    Returns ({system: {k: array}}, recording number of every tested step, mean number of candidates)."""
    n = len(V["group"])
    names = [RANDOM, PRIOR] + list(vectors)
    sums = {name: {k: np.zeros(n) for k in KS} for name in names}
    for draw in range(args.draws):
        cand = make_candidates(V, same_config, args.k, np.random.default_rng(args.seed + draw))
        n_cand = (cand >= 0).sum(axis=1)
        for k in KS:
            sums[RANDOM][k] += np.minimum(k, n_cand) / n_cand
        for k, hit in expected_top_hits(prior_scores(V, cand, prior)).items():
            sums[PRIOR][k] += hit
        for name, (zs, za) in vectors.items():
            for k, hit in top_hits(candidate_scores(zs, za, cand)).items():
                sums[name][k] += hit
    tested = (cand[:, 1:] >= 0).sum(axis=1) >= MIN_DISTRACTORS       # the same for every draw
    result = {name: {k: (sums[name][k] / args.draws)[tested] for k in KS} for name in names}
    recording = np.unique(V["group"][tested], return_inverse=True)[1]
    return result, recording, (n_cand[tested].mean() if tested.any() else 0.0)


# ====================================== INTERVALS ======================================
def bootstrap_setup(recording, n_boot, seed):
    """Random resamples of whole recordings (with replacement), reused for every number so differences stay paired."""
    n_rec = recording.max() + 1
    draws = np.random.default_rng(seed).integers(0, n_rec, size=(n_boot, n_rec))
    return draws, np.bincount(recording, minlength=n_rec).astype(float)


def interval(per_step, recording, setup):
    """(mean, low, high): the mean over steps and its 95% interval."""
    draws, sizes = setup
    totals = np.bincount(recording, weights=per_step, minlength=len(sizes))
    means = totals[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(per_step.mean()), float(low), float(high)


def cell(stats):
    mean, low, high = stats
    return f"{mean:6.1%} [{low:5.1%}, {high:5.1%}]"


def show(title, result, recording, mean_cands, args):
    setup = bootstrap_setup(recording, args.boot, args.seed)
    print(f"\n================ {title} ================")
    print(f"{len(recording)} held-out steps from {recording.max() + 1} held-out recordings, "
          f"{mean_cands:.1f} candidates per step on average, {args.draws} random draws of candidates")
    print(f"{'':28s} {'top-1 (95% interval)':>24s} {'top-3 (95% interval)':>24s}")
    for name, per_k in result.items():
        print(f"{name:28s} {cell(interval(per_k[1], recording, setup)):>24s} {cell(interval(per_k[3], recording, setup)):>24s}")

    print("\nSame steps, difference in percentage points (first minus second; the interval excludes 0 = a clear difference):")
    pairs = [("after 'post'", "after 'mid'"), ("after 'post'", PRIOR), ("after 'post'", "raw Qwen vectors"),
             ("after 'mid'", "raw Qwen vectors"), ("after 'mid'", PRIOR)]
    for a, b in pairs:
        if a in result and b in result:
            cells = []
            for k in KS:
                mean, low, high = interval(result[a][k] - result[b][k], recording, setup)
                cells.append(f"top-{k} {mean * 100:+5.1f} [{low * 100:+5.1f}, {high * 100:+5.1f}]")
            print(f"  {a:14s} - {b:28s} {cells[0]:>32s} {cells[1]:>32s}")
    if "after 'post'" in result and "after 'mid'" in result:
        mean, low, high = interval(result["after 'post'"][1] - result["after 'mid'"][1], recording, setup)
        verdict = ("stage 3 clearly HELPED on held-out recordings" if low > 0 else
                   "stage 3 clearly HURT on held-out recordings" if high < 0 else
                   "no clear difference between post and mid: stage 3 did not demonstrably learn anything here")
        print(f"\nVerdict (top-1, post vs mid): {verdict}.")


# ======================================== MAIN ========================================
@torch.no_grad()
def head_vectors(state_head, action_head, S, A, device):
    """Run the held-out state and action vectors through a pair of heads. Returns two numpy arrays."""
    zs = state_head(torch.from_numpy(S).to(device).float()).cpu().numpy()
    za = action_head(torch.from_numpy(A).to(device).float()).cpu().numpy()
    return zs, za


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=16, help="candidates per step, the true action included")
    parser.add_argument("--draws", type=int, default=5, help="random draws of the wrong candidates (averaged)")
    parser.add_argument("--boot", type=int, default=2000, help="bootstrap resamples for the 95% intervals")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"        # the heads are tiny; no backbone is needed here
    D = load_agent_data()
    val = np.flatnonzero(D["is_val"])
    train_groups, val_groups = set(D["group"][~D["is_val"]]), set(D["group"][val])
    print(f"Agent data: {int((~D['is_val']).sum())} training steps from {len(train_groups)} recordings, "
          f"{len(val)} held-out steps from {len(val_groups)} recordings.")
    if train_groups & val_groups:
        print(f"WARNING: {len(train_groups & val_groups)} recordings appear in both training and test. The split is broken; do not trust the numbers.")
    else:
        print("Split check: no recording is in both training and test.")
    if len(val) < 50:
        raise SystemExit("Too few held-out steps for a meaningful test. Rebuild with a bigger --per-config in 1b_build_agent_pairs.py.")
    common = Counter(D["type"][val]).most_common(5)
    print("Most common action types in the held-out steps: " + ", ".join(f"{t} {c / len(val):.0%}" for t, c in common))

    V = {key: D[key][val] for key in ("group", "config", "type", "text_id")}
    S, A = D["S"][val], D["A"][val]
    prior = train_prior(D)

    systems = {"raw Qwen vectors": (torch.nn.Identity(), torch.nn.Identity())}
    for stage in ("pre", "mid", "post"):
        path = clm.ckpt_path(stage)
        if os.path.exists(path):
            state_head, action_head, _ = clm.load_heads(path, device)
            systems[f"after '{stage}'"] = (state_head.eval(), action_head.eval())
        else:
            print(f"(stage '{stage}' skipped: {path} not found)")
    vectors = {name: head_vectors(sh, ah, S, A, device) for name, (sh, ah) in systems.items()}

    for title, same_config in (("SAME-CONFIG candidates (hard: the test to trust)", True),
                               ("ANY-CONFIG candidates (easy: what 4_train.py reports)", False)):
        result, recording, mean_cands = evaluate_set(V, vectors, prior, same_config, args)
        if len(recording) == 0:
            print(f"\n{title}: no step had enough wrong candidates.")
            continue
        show(title, result, recording, mean_cands, args)

    print("\nHow to read it: 'prior' is what you get by knowing only the dataset. A trained row must clearly beat it, and 'post' must "
          "clearly beat 'mid' (the last block), or stage 3 did not teach the heads anything about agent decisions.\n"
          "Caveats for the README: the split is by recording, not by task, so a task seen in training can appear again as a test "
          "recording; and the held-out set is small, so look at the intervals, not just the means.")


if __name__ == "__main__":
    main()
