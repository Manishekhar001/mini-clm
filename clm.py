"""Shared building blocks for the mini CLM: settings, the frozen backbone, the embedding
function and the small "head" network.

Every other script imports this file. If you want to change a setting, this is the place.

The whole idea in one picture:

    situation text --> frozen Qwen --> 1024 numbers --> STATE head  --> 256 numbers --+
                                                                                       +--> cosine = score
    option text    --> frozen Qwen --> 1024 numbers --> ACTION head --> 256 numbers --+

Only the two small heads are trained. Qwen itself is never changed.

HOW A TEXT BECOMES ONE VECTOR (the "recipe", see the settings below):
    1. The situation is written as plain text and ends with "Answer:" (no chat-template tokens).
    2. Qwen reads it and we AVERAGE the hidden states of all tokens (mean pooling).
    3. We subtract the average vector of the training data (centering), so what is left is what makes
       one text DIFFERENT from another.
The first version of this project took the hidden state of the LAST token of a ChatML text, which is the
"\\n" after <|im_end|>. That token only knows "a new turn starts now", so every question got almost the same
vector (cosine 0.99) and the model could not tell questions apart. Do not go back to that.
"""
import os
import zlib

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # hides a harmless Windows warning

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# ============================== SETTINGS (edit here) ==============================
MODEL_NAME = "Qwen/Qwen3-0.6B"   # the frozen backbone (hidden size 1024). Any Qwen3 model works.
MAX_STATE_TOKENS = 1024          # a situation is cut to its LAST 1024 tokens (recent context matters most)
MAX_ACTION_TOKENS = 128          # an option is cut to its FIRST 128 tokens
EMBED_BATCH = 8                  # sequences per GPU batch while embedding. Lower it if you run out of memory.
HEAD_WIDTH = 1024                # hidden width of each head
HEAD_ARCH = "residual-v1"        # bumped when the head design changes; older checkpoints are then refused
PROJ_DIM = 256                   # (not used any more: the residual head outputs a vector as big as its input)

# ---- The embedding recipe. Changing ANY of these means re-running steps 2, 2b, 3 and 4. ----
# Run 2c_compare_extraction.py to pick these by measured accuracy instead of guessing.
STATE_MODE = "plain"             # "plain" = the raw text (recommended). "chatml" / "chatml_gen" = Qwen chat template (the old, broken recipe)
STATE_SUFFIX = "\n\nAnswer:"     # added to the end of every plain-text state
ACTION_PREFIX = "Answer: "               # added to the start of every option, e.g. "Answer: ". Empty is best with mean pooling
POOLING = "mean"                 # "mean" = average of all token vectors, "last" = only the last token's vector
CENTER = True                    # subtract the training-data average vector (states and actions separately)

RECIPE = f"state={STATE_MODE}{STATE_SUFFIX!r}|action={ACTION_PREFIX!r}|pool={POOLING}|center={CENTER}"
RECIPE_ID = f"{zlib.crc32(RECIPE.encode()):08x}"   # stored with every vector file and every checkpoint; a mismatch stops the run

# ================================== FILE LOCATIONS ================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
QA_PAIRS_PATH = os.path.join(DATA_DIR, "qa_pairs.jsonl")        # written by 1a_build_qa_pairs.py
AGENT_PAIRS_PATH = os.path.join(DATA_DIR, "agent_pairs.jsonl")  # written by 1b_build_agent_pairs.py
NEG_PATH = os.path.join(DATA_DIR, "negatives.npz")              # written by 3_mine_negatives.py
SYN_TEXT_PATH = os.path.join(DATA_DIR, "synthetic_negatives.jsonl")   # written by 2b_write_negatives.py (as it goes)
SYN_NEG_PATH = os.path.join(DATA_DIR, "synthetic_negatives.npz")      # written by 2b_write_negatives.py (at the end)
CENTER_PATH = os.path.join(DATA_DIR, "centering.npz")           # written by 2_embed.py: the average state / action vector


def emb_path(kind):
    """Where the vectors for 'qa' or 'agent' pairs live (written by 2_embed.py)."""
    return os.path.join(DATA_DIR, f"embeddings_{kind}.npz")


def ckpt_path(stage):
    """Where the trained heads of a stage ('pre', 'mid' or 'post') live (written by 4_train.py)."""
    return os.path.join(DATA_DIR, f"heads_{stage}.pt")


# ============================ SMALL HELPERS FOR Q&A PAIRS ============================
def question_of(pair):
    """The question of a Q&A pair: the text after the last "Question:" in its state."""
    return pair["state"][-1]["content"].split("Question:")[-1].strip()


def pair_key(pair):
    """A number that identifies a Q&A pair by its question and answer text. Files made for a pair store it, so a rebuilt
    qa_pairs.jsonl is never matched with extras that belonged to the old pairs."""
    return zlib.crc32((question_of(pair) + "\x00" + pair["action"]).encode())


# ============================ TEXT -> TOKEN IDS (the recipe) ============================
def render_plain(kept):
    """A conversation as plain text. One message = just its text. Several = "role: text" blocks."""
    if len(kept) == 1:
        return kept[0]["content"]
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in kept)


def state_to_ids(tok, messages, mode=None, suffix=None):
    """Turn a conversation (list of {"role", "content"}) into token ids, keeping the END of it.

    mode / suffix default to STATE_MODE / STATE_SUFFIX. (2c_compare_extraction.py passes other values to compare them.)"""
    mode = STATE_MODE if mode is None else mode
    suffix = STATE_SUFFIX if suffix is None else suffix
    kept, chars = [], 0
    for msg in reversed(messages):               # walk backwards: newest messages first
        kept.append(msg)
        chars += len(msg["content"])
        if chars > 8 * MAX_STATE_TOKENS:         # plenty of text; no need to tokenize more
            break
    kept.reverse()
    text = None
    if mode != "plain":                          # the old recipe: Qwen's chat template
        try:
            text = tok.apply_chat_template(kept, tokenize=False, add_generation_prompt=(mode == "chatml_gen"))
        except Exception:                        # some message orders break the template; fall back to plain text
            text = None
    if text is None:
        text = render_plain(kept) + (suffix if mode == "plain" else "")
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return ids[-MAX_STATE_TOKENS:]


def action_to_ids(tok, text, prefix=None):
    """Turn an option (plain text) into token ids, keeping the START of it."""
    prefix = ACTION_PREFIX if prefix is None else prefix
    ids = tok(prefix + (text or " "), add_special_tokens=False)["input_ids"]
    return ids[:MAX_ACTION_TOKENS] or [tok.pad_token_id]


# ================================== THE BACKBONE ==================================
def get_device():
    """Return "cuda". Stops with a clear message if PyTorch cannot see the GPU (it never silently uses the CPU)."""
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch cannot see a CUDA GPU (torch.cuda.is_available() is False), so this would run on the CPU.\n"
                         "Most likely the CPU-only PyTorch is installed. See README > Troubleshooting: reinstall with the --index-url command.")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} ({props.total_memory / 1024 ** 3:.1f} GB)")
    return "cuda"


def load_backbone():
    """Load Qwen (without its text-writing layer) on the GPU, frozen. Returns (tokenizer, model)."""
    device = get_device()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading {MODEL_NAME} on {device} ({dtype}) ...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)                  # frozen: we never train the backbone
    return tok, model


@torch.inference_mode()
def embed_all(model, tok, token_lists, modes=("mean", "last"), batch_size=EMBED_BATCH):
    """Run token lists through Qwen once and return {mode: one unit-length vector per list (float16, N x hidden)}.

    mode "mean" = average of the hidden states of all real tokens (padding is ignored).
    mode "last" = the hidden state of the last token only."""
    device = next(model.parameters()).device
    pad_id = tok.pad_token_id
    # Longest first: padding is minimal, and an out-of-memory error shows up immediately, not an hour in.
    order = np.argsort([-len(t) for t in token_lists])
    out = {m: np.zeros((len(token_lists), model.config.hidden_size), dtype=np.float16) for m in modes}

    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        longest = max(len(token_lists[i]) for i in idx)
        ids = torch.full((len(idx), longest), pad_id, dtype=torch.long)
        mask = torch.zeros((len(idx), longest), dtype=torch.long)
        for row, i in enumerate(idx):
            t = token_lists[i]
            ids[row, longest - len(t):] = torch.tensor(t)   # pad on the LEFT so the last token is always real
            mask[row, longest - len(t):] = 1
        mask = mask.to(device)
        hidden = model(input_ids=ids.to(device), attention_mask=mask, use_cache=False).last_hidden_state.float()   # no cache: saves GPU memory
        weights = mask.unsqueeze(-1).float()
        for m in modes:
            if m == "last":
                vec = hidden[:, -1]                                           # hidden state of the last token
            elif m == "mean":
                vec = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)   # average over real tokens only
            else:
                raise ValueError(f"unknown pooling '{m}' (use 'mean' or 'last')")
            vec = vec / (vec.norm(dim=-1, keepdim=True) + 1e-12)              # unit length
            out[m][idx] = vec.cpu().numpy().astype(np.float16)

    for m in modes:
        if not np.isfinite(out[m].astype(np.float32)).all():
            raise RuntimeError("Got NaN/inf in the embeddings. Try a smaller EMBED_BATCH or float32 (see README).")
    return out


def embed(model, tok, token_lists, batch_size=EMBED_BATCH, pooling=None):
    """One unit-length vector per token list (float16, N x hidden), pooled as POOLING says. NOT centered yet (see center_*)."""
    pooling = POOLING if pooling is None else pooling
    return embed_all(model, tok, token_lists, (pooling,), batch_size)[pooling]


# ================================== CENTERING ==================================
# LLM hidden states all share a big common component, so raw vectors of different texts point almost the same way.
# Subtracting the average vector removes it. The average is computed ONCE from the Q&A training vectors (2_embed.py)
# and then applied to everything: Q&A, agent data, written negatives and the texts given to the demo.
_CENTER_CACHE = {}


def mean_pairwise_cosine(vecs, n=1000, seed=0):
    """Average cosine between DIFFERENT texts (sampled). Near 1.0 = every text looks the same = collapsed."""
    v = np.asarray(vecs)
    if len(v) > n:                                   # sample first, so a 90,000-row array is not converted whole
        v = v[np.random.default_rng(seed).choice(len(v), n, replace=False)]
    v = v.astype(np.float32)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    g = v @ v.T
    k = len(v)
    return float((g.sum() - np.trace(g)) / max(k * (k - 1), 1))


def center_vectors(vecs, mean):
    """(vecs - mean), back to unit length, float16."""
    x = np.asarray(vecs, dtype=np.float32) - np.asarray(mean, dtype=np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x.astype(np.float16)


def save_centering(state_mean, action_mean):
    np.savez(CENTER_PATH, state_mean=np.asarray(state_mean, dtype=np.float32),
             action_mean=np.asarray(action_mean, dtype=np.float32), recipe=np.array(RECIPE_ID))
    _CENTER_CACHE.clear()


def load_centering():
    """(state_mean, action_mean) saved by 2_embed.py."""
    if "c" not in _CENTER_CACHE:
        if not os.path.exists(CENTER_PATH):
            raise SystemExit(f"{CENTER_PATH} not found. Run `python 2_embed.py --only qa` first (it creates it).")
        c = np.load(CENTER_PATH)
        if str(c["recipe"]) != RECIPE_ID:
            raise SystemExit("centering.npz was made with a different embedding recipe than clm.py says now. Rerun step 2.")
        _CENTER_CACHE["c"] = (c["state_mean"], c["action_mean"])
    return _CENTER_CACHE["c"]


def center_saved(vecs, which):
    """Center vecs with the SAVED mean. which = "state" or "action". Does nothing when CENTER is False."""
    if not CENTER:
        return vecs
    state_mean, action_mean = load_centering()
    return center_vectors(vecs, state_mean if which == "state" else action_mean)


# ================================== THE TRAINABLE HEAD ==================================
class Head(nn.Module):
    """A small network that gently adjusts a Qwen vector. Used twice: state head, action head.

        output = unit_length( x + W2 * GELU( W1 * x ) )

    The last layer W2 starts at ZERO, so before training the head does nothing and the model is exactly as good as the
    raw Qwen vectors. Training can only add corrections on top of what Qwen already knows (a "residual" head).
    The old head had three layers and rebuilt the whole space from scratch; it squeezed all answers of one kind
    (all author names, say) into one spot and made the model worse than raw Qwen.
    """

    def __init__(self, in_dim, width=HEAD_WIDTH):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, width)
        self.out = nn.Linear(width, in_dim)
        self.act = nn.GELU()
        nn.init.zeros_(self.out.weight)       # start as "change nothing"
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        x = x + self.out(self.act(self.layer1(x)))
        return nn.functional.normalize(x, dim=-1)   # unit length, so dot product = cosine


def save_heads(path, state_head, action_head, log_scale, in_dim, stage, metrics):
    """Save a trained pair of heads (plus the learned temperature) to one file."""
    torch.save({"state_head": {k: v.cpu() for k, v in state_head.state_dict().items()},
                "action_head": {k: v.cpu() for k, v in action_head.state_dict().items()},
                "log_scale": log_scale.detach().cpu(),
                "in_dim": in_dim, "width": HEAD_WIDTH, "arch": HEAD_ARCH,
                "model_name": MODEL_NAME, "recipe": RECIPE_ID, "stage": stage, "metrics": metrics}, path)


def load_heads(path, device):
    """Load saved heads. Returns (state_head, action_head, log_scale).  scale = log_scale.exp().clamp(max=100)"""
    ck = torch.load(path, map_location=device, weights_only=True)
    if ck["model_name"] != MODEL_NAME:
        raise SystemExit(f"{path} was trained on {ck['model_name']} but clm.py says {MODEL_NAME}.")
    if ck.get("recipe") != RECIPE_ID:
        raise SystemExit(f"{path} was trained with a different (older) embedding recipe than clm.py says now, so it would give "
                         "wrong answers. Rerun steps 2, 3 and `python 4_train.py --stage all`.")
    if ck.get("arch") != HEAD_ARCH:
        raise SystemExit(f"{path} was trained with an older head design than clm.py has now. Run `python 4_train.py --stage all` again.")
    state_head = Head(ck["in_dim"], ck["width"]).to(device)
    action_head = Head(ck["in_dim"], ck["width"]).to(device)
    state_head.load_state_dict(ck["state_head"])
    action_head.load_state_dict(ck["action_head"])
    return state_head, action_head, ck["log_scale"].to(device)


# ===================================== INFERENCE =====================================
class Scorer:
    """Use the trained model. Give it a situation and some options; get a probability for each option.

        import clm
        scorer = clm.Scorer()                                   # loads Qwen + the heads of every stage that was trained
        scorer.rank("Question: Capital of Japan?", ["Tokyo", "Paris"])
        # -> {"pre": [0.8, 0.2], "mid": [0.9, 0.1], "post": [0.9, 0.1]}

    Each stage's heads give their own answer, so you can see what each training stage learned.
    """

    def __init__(self, stages=("pre", "mid", "post")):
        self.tok, self.model = load_backbone()
        self.device = next(self.model.parameters()).device
        self.heads = {}                                         # stage -> (state head, action head, scale)
        for stage in stages:
            path = ckpt_path(stage)
            if not os.path.exists(path):
                print(f"(stage '{stage}' skipped: {path} not found. Run: python 4_train.py --stage {stage})")
                continue
            state_head, action_head, log_scale = load_heads(path, self.device)
            self.heads[stage] = (state_head.eval(), action_head.eval(), log_scale.exp().clamp(max=100.0))
        if not self.heads:
            raise SystemExit("No trained heads found. Run `python 4_train.py --stage all` first.")

    def _vectors(self, token_lists, which):
        """Embed with the SAME recipe as training: pooling, then the saved centering."""
        vecs = center_saved(embed(self.model, self.tok, token_lists), which)
        return torch.from_numpy(vecs).float().to(self.device)

    @torch.inference_mode()
    def rank(self, state, options):
        """state: text (or a list of {"role", "content"} messages). options: list of texts.
        Returns {stage: [probability of each option, in the order given]}. Each list adds up to 1."""
        if isinstance(state, str):
            state = [{"role": "user", "content": state}]
        state_vec = self._vectors([state_to_ids(self.tok, state)], "state")
        option_vecs = self._vectors([action_to_ids(self.tok, o) for o in options], "action")
        result = {}
        for stage, (state_head, action_head, scale) in self.heads.items():
            scores = scale * (state_head(state_vec) @ action_head(option_vecs).T)   # cosine x scale
            result[stage] = torch.softmax(scores, dim=-1)[0].tolist()                # one probability per option
        return result
