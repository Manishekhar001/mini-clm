"""Step 4 - train the two heads (state head + action head) in THREE stages.

    stage 1  pre   Q&A pairs, plus 3 hard negatives per question. Each question must score highest with its own answer.
    stage 2  mid   Same Q&A pairs, plus up to 10 "hard negatives" per question (plausible wrong answers).
    stage 3  post  60% agent decisions + 40% Q&A replay (with hard negatives). Temperature frozen.

Each stage starts from the heads the previous stage saved. The Qwen vectors are already computed
(step 2), so we only train two tiny networks and every stage takes minutes, not days.

The heads are "residual": they start as a do-nothing copy of the raw Qwen vectors and only learn small corrections
(see clm.Head). Learning rates are low on purpose so training cannot scramble what Qwen already knows.

The loss in plain words (InfoNCE): in a batch of N pairs, score every state against every action.
Row i should peak at action i. We check this in both directions (state -> action, action -> state).

    score(state, action) = scale * cosine(state_head(state), action_head(action))

Run:
    python 4_train.py --stage all          # all three stages, then a results table
    python 4_train.py --stage pre          # or one stage at a time: pre, mid, post
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import clm

PRE_NEG = 3           # hard negatives per question in stage 1 (stage 2 uses all NUM_NEG). Random wrong answers alone are too easy.
MAX_VAL = 2048        # test pairs used per dataset
NUM_NEG = 10          # hard negatives per Q&A pair (set in 3_mine_negatives.py)

# What differs between the stages. Everything else is shared.
#   lr:      learning rate. None = a formula that grows with the batch size: 2e-3 * sqrt(1024/width) * sqrt(batch/1024)
#   pick:    the test number (lower is better) used to choose the best epoch to keep
#   patience: stop after this many epochs without improvement
STAGES = {
    "pre":  {"init": None,  "lr": 3e-4,
             "epochs": 30, "patience": 5, "learn_scale": True, "pick": "tricky_loss",   # NOT qa_loss: random wrong answers are too easy and hid the original bug
            
             "what": "Q&A pairs + a few hard negatives each (warm-up)"},
    "mid":  {"init": "pre", "lr": 1e-4,
             "epochs": 12, "patience": 3, "learn_scale": True, "pick": "tricky_loss",
             "what": "Q&A pairs + hard negatives"},
    "post": {"init": "mid", "lr": 5e-5,
             "epochs": 12, "patience": 3, "learn_scale": False, "pick": "agent_loss",
             "what": "60% agent decisions + 40% Q&A replay with hard negatives, temperature frozen"},
}


# ======================================= DATA =======================================
def load_data(device):
    """Load all vectors onto the GPU (they are small) and remember which pairs are for training vs testing."""
    for kind, script in (("qa", "1a_build_qa_pairs.py"), ("agent", "1b_build_agent_pairs.py")):
        if not os.path.exists(clm.emb_path(kind)):
            raise SystemExit(f"{clm.emb_path(kind)} not found. Run {script} and then 2_embed.py.")
    if not os.path.exists(clm.NEG_PATH):
        raise SystemExit("negatives.npz not found. Run 3_mine_negatives.py.")

    D = {}
    for kind in ("qa", "agent"):
        d = np.load(clm.emb_path(kind))
        if "recipe" not in d.files or str(d["recipe"]) != clm.RECIPE_ID:
            raise SystemExit(f"{kind} vectors were made with a different (older) embedding recipe than clm.py says now. Rerun step 2 (python 2_embed.py), then 3.")
        if str(d["model_name"]) != clm.MODEL_NAME:
            raise SystemExit(f"{kind} vectors were made with {d['model_name']} but clm.py says {clm.MODEL_NAME}. Rerun step 2.")
        D[f"{kind}_s"] = torch.from_numpy(d["states"]).to(device)       # state vectors  (float16)
        D[f"{kind}_a"] = torch.from_numpy(d["actions"]).to(device)      # action vectors (float16)
        D[f"{kind}_g"] = torch.from_numpy(d["group"]).to(device)        # which document / recording each pair is from
        is_val = d["is_val"]
        D[f"{kind}_train"] = torch.from_numpy(np.flatnonzero(~is_val)).to(device)
        D[f"{kind}_val"] = torch.from_numpy(np.flatnonzero(is_val)[:MAX_VAL]).to(device)
        if len(D[f"{kind}_val"]) < 64 or len(D[f"{kind}_train"]) < 256:
            raise SystemExit(f"Too little {kind} data ({len(D[f'{kind}_train'])} train, {len(D[f'{kind}_val'])} test). "
                             "Rebuild with a bigger --max-pairs / --per-config.")
        if torch.unique(D[f"{kind}_g"][D[f"{kind}_val"]]).numel() < 20:   # the 16-option test needs wrong answers from other groups
            raise SystemExit(f"The {kind} test pairs come from fewer than 20 documents/recordings. "
                             "Rebuild with a bigger --max-pairs / --per-config.")
    def _collapse(x):                                                # mean cosine between different texts, on a sample
        return clm.mean_pairwise_cosine(x[torch.randperm(len(x), device=x.device)[:1000]].float().cpu().numpy())
    cs, ca = _collapse(D["qa_s"]), _collapse(D["qa_a"])
    print(f"Collapse check (mean cosine between different texts; near 1.0 = all texts look the same): states {cs:.3f}, actions {ca:.3f}")
    if cs > 0.9:
        print("WARNING: the state vectors are almost identical to each other, so the model cannot tell questions apart. "
              "Check the recipe in clm.py and rerun step 2.")
    neg = np.load(clm.NEG_PATH)
    D["qa_neg"] = torch.from_numpy(neg["neg_idx"]).to(device)   # (N, 10) indices, -1 = none. i < N: answer of Q&A pair i. i >= N: a written negative
    extra = torch.from_numpy(neg["extra"]).to(device) if "extra" in neg.files else D["qa_a"][:0]
    D["neg_pool"] = torch.cat([D["qa_a"], extra])                 # every answer vector a negative can point to
    if len(D["qa_neg"]) != len(D["qa_s"]):
        raise SystemExit("negatives.npz does not match the Q&A vectors. Rerun 3_mine_negatives.py.")
    print(f"Q&A: {len(D['qa_train'])} train / {len(D['qa_val'])} test pairs.  "
          f"Agent: {len(D['agent_train'])} train / {len(D['agent_val'])} test pairs.")
    return D


# ================================== LOSS AND TESTS ==================================
def neg_vectors(action_head, qa_actions, neg_rows):
    """Run each pair's hard-negative answers through the action head.

    neg_rows: (B, 10) indices into the Q&A pairs (-1 = no negative in this slot).
    Returns the vectors (B, 10, d) and a True/False mask (B, 10) saying which slots are real."""
    valid = neg_rows >= 0
    raw = qa_actions[neg_rows.clamp(min=0)].float()                  # (B, 10, hidden); -1 slots read pair 0 and are masked later
    vecs = action_head(raw.reshape(-1, raw.shape[-1])).reshape(*neg_rows.shape, -1)
    return vecs, valid


def contrastive_loss(state_vecs, action_vecs, scale, group, neg_vecs=None, neg_valid=None):
    """Symmetric InfoNCE, optionally with hard negatives added to the state -> action direction."""
    logits = scale * state_vecs @ action_vecs.T                      # N x N table of scores
    same = group[:, None] == group[None, :]                          # pairs from the same document / recording ...
    same.fill_diagonal_(False)                                       # ... are not counted as wrong answers
    logits = logits.masked_fill(same, float("-inf"))
    labels = torch.arange(len(logits), device=logits.device)         # the right answer for row i is column i

    forward_logits = logits
    if neg_vecs is not None:                                         # extra wrong answers for each state
        neg_logits = scale * torch.einsum("bd,bkd->bk", state_vecs, neg_vecs)
        neg_logits = neg_logits.masked_fill(~neg_valid, float("-inf"))
        forward_logits = torch.cat([logits, neg_logits], dim=1)
    state_to_action = F.cross_entropy(forward_logits, labels)        # "which answer matches this question?"
    action_to_state = F.cross_entropy(logits.T, labels)              # "which question matches this answer?"
    return (state_to_action + action_to_state) / 2


@torch.no_grad()
def choose_accuracy(state_vecs, action_vecs, group, n_options, seed=0):
    """Show each state with its true action hidden among n_options-1 random actions from other
    documents / recordings. How often is the true one scored highest? (random guessing = 1/n_options)"""
    rng = np.random.default_rng(seed)
    group_np = group.cpu().numpy()
    options = np.zeros((len(group_np), n_options), dtype=np.int64)   # column 0 = the true action
    for i in range(len(group_np)):
        others = np.flatnonzero(group_np != group_np[i])
        options[i, 0] = i
        options[i, 1:] = rng.choice(others, n_options - 1, replace=False)
    options = torch.from_numpy(options).to(state_vecs.device)
    scores = torch.einsum("nd,nkd->nk", state_vecs, action_vecs[options])
    return (scores.argmax(dim=1) == 0).float().mean().item()


@torch.no_grad()
def evaluate(state_head, action_head, scale, D, n_options=16):
    """The same three tests for every stage, so the stages can be compared:
         qa     pick the right answer among 16 (Q&A test pairs)
         tricky pick the right answer among the correct one + its hard negatives
         agent  pick the right action among 16 (agent test pairs)"""
    m = {}
    idx = D["qa_val"]
    zs, za, g = state_head(D["qa_s"][idx].float()), action_head(D["qa_a"][idx].float()), D["qa_g"][idx]
    m["qa_loss"] = contrastive_loss(zs, za, scale, g).item()
    m["qa_acc"] = choose_accuracy(zs, za, g, n_options)
    n_s = len(zs)
    m["state_cos"] = ((zs @ zs.T).sum() - n_s).item() / max(n_s * (n_s - 1), 1)   # mean cosine between different states (unit vectors); ~1 = collapsed

    zn, valid = neg_vectors(action_head, D["neg_pool"], D["qa_neg"][idx])
    has = valid.any(dim=1)                                           # only pairs that have at least one negative
    if not has.any():
        raise SystemExit("No hard negatives for the test pairs. Rerun 3_mine_negatives.py.")
    pos = (zs * za).sum(-1, keepdim=True)
    neg = torch.einsum("bd,bkd->bk", zs, zn).masked_fill(~valid, float("-inf"))
    m["tricky_acc"] = (pos[has, 0] > neg[has].max(dim=1).values).float().mean().item()
    logits = scale * torch.cat([pos, neg], dim=1)[has]
    m["tricky_loss"] = F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=logits.device)).item()
    m["tricky_chance"] = (1.0 / (1 + valid[has].sum(dim=1))).float().mean().item()

    idx = D["agent_val"]
    zs, za, g = state_head(D["agent_s"][idx].float()), action_head(D["agent_a"][idx].float()), D["agent_g"][idx]
    m["agent_loss"] = contrastive_loss(zs, za, scale, g).item()
    m["agent_acc"] = choose_accuracy(zs, za, g, n_options)
    return m


# =================================== TRAINING ===================================
def train_stage(stage, args, D, device):
    cfg = STAGES[stage]
    batch = min(args.batch, len(D["qa_train"]))
    lr = args.lr or cfg["lr"] or 2e-3 * math.sqrt(1024 / clm.HEAD_WIDTH) * math.sqrt(batch / 1024)
    epochs = args.epochs or cfg["epochs"]
    in_dim = D["qa_s"].shape[1]

    if cfg["init"]:                                                  # continue from the previous stage
        init_path = clm.ckpt_path(cfg["init"])
        if not os.path.exists(init_path):
            raise SystemExit(f"Stage '{stage}' continues from stage '{cfg['init']}'. Run `python 4_train.py --stage {cfg['init']}` first.")
        state_head, action_head, loaded = clm.load_heads(init_path, device)
        log_scale = torch.nn.Parameter(loaded.clone())
    else:
        state_head, action_head = clm.Head(in_dim).to(device), clm.Head(in_dim).to(device)
        log_scale = torch.nn.Parameter(torch.tensor(math.log(1 / 0.07), device=device))   # learned "temperature", starts at 0.07
    log_scale.requires_grad_(cfg["learn_scale"])
    params = list(state_head.parameters()) + list(action_head.parameters()) + ([log_scale] if cfg["learn_scale"] else [])
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    if stage == "post":                                              # each step: 60% agent pairs + 40% Q&A replay
        main_pool = D["agent_train"]
        per_step = min(int(batch * 0.6), len(main_pool))
        n_replay = batch - per_step
    else:
        main_pool, per_step, n_replay = D["qa_train"], batch, 0

    print(f"\n=== Stage '{stage}': {cfg['what']} ===")
    print(f"batch {batch}, learning rate {lr:.5f}, up to {epochs} epochs, "
          f"temperature {'learned' if cfg['learn_scale'] else 'frozen'}")
    steps = len(main_pool) // per_step
    print(f"{len(main_pool)} training pairs, {per_step} per update -> {steps} updates per epoch "
          f"(each epoch uses {steps * per_step} of them; the others are picked differently every epoch)")
    best, best_epoch, t0 = float("inf"), 0, time.time()

    for epoch in range(1, epochs + 1):
        order = main_pool[torch.randperm(len(main_pool), device=device)]
        losses = []
        for j in range(0, len(order) - per_step + 1, per_step):
            rows = order[j:j + per_step]
            if stage == "pre":
                s, a, g, neg_rows = D["qa_s"][rows], D["qa_a"][rows], D["qa_g"][rows], D["qa_neg"][rows][:, :PRE_NEG]
            elif stage == "mid":
                s, a, g, neg_rows = D["qa_s"][rows], D["qa_a"][rows], D["qa_g"][rows], D["qa_neg"][rows]
            else:
                replay = D["qa_train"][torch.randint(len(D["qa_train"]), (n_replay,), device=device)]
                s = torch.cat([D["qa_s"][replay], D["agent_s"][rows]])
                a = torch.cat([D["qa_a"][replay], D["agent_a"][rows]])
                g = torch.cat([D["qa_g"][replay], D["agent_g"][rows]])
                no_negs = torch.full((len(rows), NUM_NEG), -1, dtype=torch.long, device=device)   # agent pairs have no negatives
                neg_rows = torch.cat([D["qa_neg"][replay], no_negs])

            state_vecs, action_vecs = state_head(s.float()), action_head(a.float())
            scale = log_scale.exp().clamp(max=100.0)
            neg_vecs = neg_valid = None
            if neg_rows is not None:
                neg_vecs, neg_valid = neg_vectors(action_head, D["neg_pool"], neg_rows)
            loss = contrastive_loss(state_vecs, action_vecs, scale, g, neg_vecs, neg_valid)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            losses.append(loss.item())

        if not math.isfinite(float(np.mean(losses))):
            raise SystemExit("The loss became NaN. Try a smaller --lr.")
        m = evaluate(state_head, action_head, log_scale.detach().exp().clamp(max=100.0), D)
        note = ""
        if m[cfg["pick"]] < best - 1e-4:
            best, best_epoch, note = m[cfg["pick"]], epoch, "  <- best, saved"
            clm.save_heads(clm.ckpt_path(stage), state_head, action_head, log_scale, in_dim, stage, m)
        print(f"epoch {epoch:2d}  train loss {np.mean(losses):6.3f} | test: Q&A {m['qa_acc']:5.1%}  "
              f"tricky {m['tricky_acc']:5.1%}  agent {m['agent_acc']:5.1%} | "
              f"temp {log_scale.detach().exp().clamp(max=100.0).item():5.1f}  state-cos {m['state_cos']:.2f}{note}")
        if m["state_cos"] > 0.9:
            print("  WARNING: the state head maps different questions to nearly the same vector (state-cos > 0.9). "
                  "That is the collapse bug: do not trust this model.")
        if epoch - best_epoch >= cfg["patience"]:
            print(f"No improvement in '{cfg['pick']}' for {cfg['patience']} epochs - stopping early.")
            break
    print(f"Stage '{stage}' done in {(time.time() - t0) / 60:.1f} min; kept epoch {best_epoch} -> {clm.ckpt_path(stage)}")
    if best_epoch == epochs:                                         # the last epoch was the best one: it had not finished learning
        print(f"Note: stage '{stage}' was still improving at the last epoch. For a better result run it longer, e.g. "
              f"`python 4_train.py --stage {stage} --epochs {epochs * 2}`, then rerun the stages after it.")


def report(D, device):
    """One table with every finished stage on the same tests."""
    rows = []
    identity = torch.nn.Identity()                                   # "heads" that do nothing = raw Qwen vectors
    rows.append(("raw Qwen vectors", evaluate(identity, identity, torch.tensor(1.0, device=device), D)))
    for stage in STAGES:
        if os.path.exists(clm.ckpt_path(stage)):
            sh, ah, log_scale = clm.load_heads(clm.ckpt_path(stage), device)
            rows.append((f"after '{stage}'", evaluate(sh, ah, log_scale.exp().clamp(max=100.0), D)))
    chance = rows[0][1]["tricky_chance"]
    print("\n================ RESULTS (all on test pairs the model never trained on) ================")
    print(f"{'':20s} {'Q&A (1 of 16)':>14s} {'Tricky Q&A':>12s} {'Agent (1 of 16)':>16s}")
    print(f"{'random guessing':20s} {1 / 16:14.1%} {chance:12.1%} {1 / 16:16.1%}")
    for name, m in rows:
        print(f"{name:20s} {m['qa_acc']:14.1%} {m['tricky_acc']:12.1%} {m['agent_acc']:16.1%}")
    print("Tricky Q&A = the right answer against its hard negatives. Agent = pick the right agent action.")
    print("Trust 'Tricky Q&A'. 'Q&A (1 of 16)' uses random wrong answers (a name vs a date), which are easy and hid the original bug.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pre", "mid", "post", "all"], default="all")
    parser.add_argument("--batch", type=int, default=1024, help="pairs per step; bigger = more wrong answers to learn from per step")
    parser.add_argument("--lr", type=float, default=None, help="override the learning rate of the stage")
    parser.add_argument("--epochs", type=int, default=None, help="override the maximum number of epochs")
    args = parser.parse_args()

    torch.manual_seed(1234)                                          # fixed seed, so runs are repeatable
    device = clm.get_device()
    D = load_data(device)
    for stage in (list(STAGES) if args.stage == "all" else [args.stage]):
        train_stage(stage, args, D, device)
    report(D, device)
    print("\nNext: python 5_demo.py")


if __name__ == "__main__":
    main()
