"""Shared building blocks for the mini CLM: settings, the frozen backbone, the embedding
function and the small "head" network.

Every other script imports this file. If you want to change a setting, this is the place.

The whole idea in one picture:

    situation text --> frozen Qwen --> 1024 numbers --> STATE head  --> 256 numbers --+
                                                                                       +--> cosine = score
    option text    --> frozen Qwen --> 1024 numbers --> ACTION head --> 256 numbers --+

Only the two small heads are trained. Qwen itself is never changed.
"""
import os

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
PROJ_DIM = 256                   # size of the final vector each head outputs

# ================================== FILE LOCATIONS ================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
QA_PAIRS_PATH = os.path.join(DATA_DIR, "qa_pairs.jsonl")        # written by 1a_build_qa_pairs.py
AGENT_PAIRS_PATH = os.path.join(DATA_DIR, "agent_pairs.jsonl")  # written by 1b_build_agent_pairs.py
NEG_PATH = os.path.join(DATA_DIR, "negatives.npz")              # written by 3_mine_negatives.py


def emb_path(kind):
    """Where the vectors for 'qa' or 'agent' pairs live (written by 2_embed.py)."""
    return os.path.join(DATA_DIR, f"embeddings_{kind}.npz")


def ckpt_path(stage):
    """Where the trained heads of a stage ('pre', 'mid' or 'post') live (written by 4_train.py)."""
    return os.path.join(DATA_DIR, f"heads_{stage}.pt")


# ============================ TEXT -> TOKEN IDS (the recipe) ============================
def state_to_ids(tok, messages):
    """Turn a conversation (list of {"role", "content"}) into token ids, keeping the END of it."""
    kept, chars = [], 0
    for msg in reversed(messages):               # walk backwards: newest messages first
        kept.append(msg)
        chars += len(msg["content"])
        if chars > 8 * MAX_STATE_TOKENS:         # plenty of text; no need to tokenize more
            break
    kept.reverse()
    try:
        text = tok.apply_chat_template(kept, tokenize=False, add_generation_prompt=False)
    except Exception:                            # some message orders break the template; fall back to plain text
        text = "\n\n".join(m["content"] for m in kept)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return ids[-MAX_STATE_TOKENS:]


def action_to_ids(tok, text):
    """Turn an option (plain text) into token ids, keeping the START of it."""
    ids = tok(text or " ", add_special_tokens=False)["input_ids"]
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
def embed(model, tok, token_lists, batch_size=EMBED_BATCH):
    """Run token lists through Qwen and return one unit-length vector per list (float16, N x hidden).

    The vector is the hidden state of the LAST token, which has "read" the whole text.
    """
    device = next(model.parameters()).device
    pad_id = tok.pad_token_id
    # Longest first: padding is minimal, and an out-of-memory error shows up immediately, not an hour in.
    order = np.argsort([-len(t) for t in token_lists])
    out = np.zeros((len(token_lists), model.config.hidden_size), dtype=np.float16)

    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        longest = max(len(token_lists[i]) for i in idx)
        ids = torch.full((len(idx), longest), pad_id, dtype=torch.long)
        mask = torch.zeros((len(idx), longest), dtype=torch.long)
        for row, i in enumerate(idx):
            t = token_lists[i]
            ids[row, longest - len(t):] = torch.tensor(t)   # pad on the LEFT so the last token is always real
            mask[row, longest - len(t):] = 1
        hidden = model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False).last_hidden_state   # no cache: saves GPU memory
        last = hidden[:, -1].float()                        # hidden state of the last token
        last = last / (last.norm(dim=-1, keepdim=True) + 1e-12)   # unit length
        out[idx] = last.cpu().numpy().astype(np.float16)

    if not np.isfinite(out.astype(np.float32)).all():
        raise RuntimeError("Got NaN/inf in the embeddings. Try a smaller EMBED_BATCH or float32 (see README).")
    return out


# ================================== THE TRAINABLE HEAD ==================================
class Head(nn.Module):
    """A small network that maps a Qwen vector to a shorter vector. Used twice: state head, action head.

        in_dim -> WIDTH (GELU) -> WIDTH (LayerNorm, GELU) -> PROJ_DIM -> unit length
    """

    def __init__(self, in_dim, width=HEAD_WIDTH, out_dim=PROJ_DIM):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, width)
        self.layer2 = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, out_dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.layer1(x))
        x = self.act(self.norm(self.layer2(x)))
        return nn.functional.normalize(self.out(x), dim=-1)   # unit length, so dot product = cosine


def save_heads(path, state_head, action_head, log_scale, in_dim, stage, metrics):
    """Save a trained pair of heads (plus the learned temperature) to one file."""
    torch.save({"state_head": {k: v.cpu() for k, v in state_head.state_dict().items()},
                "action_head": {k: v.cpu() for k, v in action_head.state_dict().items()},
                "log_scale": log_scale.detach().cpu(),
                "in_dim": in_dim, "width": HEAD_WIDTH, "out_dim": PROJ_DIM,
                "model_name": MODEL_NAME, "stage": stage, "metrics": metrics}, path)


def load_heads(path, device):
    """Load saved heads. Returns (state_head, action_head, log_scale).  scale = log_scale.exp().clamp(max=100)"""
    ck = torch.load(path, map_location=device, weights_only=True)
    if ck["model_name"] != MODEL_NAME:
        raise SystemExit(f"{path} was trained on {ck['model_name']} but clm.py says {MODEL_NAME}.")
    state_head = Head(ck["in_dim"], ck["width"], ck["out_dim"]).to(device)
    action_head = Head(ck["in_dim"], ck["width"], ck["out_dim"]).to(device)
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

    def _vectors(self, token_lists):
        return torch.from_numpy(embed(self.model, self.tok, token_lists)).float().to(self.device)

    @torch.inference_mode()
    def rank(self, state, options):
        """state: text (or a list of {"role", "content"} messages). options: list of texts.
        Returns {stage: [probability of each option, in the order given]}. Each list adds up to 1."""
        if isinstance(state, str):
            state = [{"role": "user", "content": state}]
        state_vec = self._vectors([state_to_ids(self.tok, state)])
        option_vecs = self._vectors([action_to_ids(self.tok, o) for o in options])
        result = {}
        for stage, (state_head, action_head, scale) in self.heads.items():
            scores = scale * (state_head(state_vec) @ action_head(option_vecs).T)   # cosine x scale
            result[stage] = torch.softmax(scores, dim=-1)[0].tolist()                # one probability per option
        return result
