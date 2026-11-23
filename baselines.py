"""Baselines (optional) - two ready-made ways to answer the same multiple-choice questions, so the CLM has something to be compared with.

Used by benchmark.py (accuracy rows) and bench_latency.py (speed rows). Nothing in this file is trained.

1. Qwen log-likelihood. The usual way to score a small language model on multiple choice, with the SAME Qwen3-0.6B the
   CLM uses as its backbone. For every option Qwen reads "situation + option" and we add up the log-probability of the
   option's tokens. That is one forward pass PER option (the CLM needs one pass for the situation, plus short ones for
   the options). Two ways to pick the winner, as in lm-eval:
       acc        the option with the highest total log-probability
       acc_norm   the highest total divided by the option's length in characters (stops short options always winning)
   It reads exactly the same situation text as the CLM (same cut to the last 1024 tokens, same "Answer:" ending), so
   the two methods see identical input.

2. Off-the-shelf embedders (bge-small-en-v1.5 and all-MiniLM-L6-v2). A small ready-made sentence-embedding model used
   as it is: embed the situation, embed every option, pick the option with the highest cosine. No training at all.
   If one of them beats the trained CLM heads, the README must say so.
   Long situations are cut from the LEFT (the end is kept), because the question is at the END of the text.
   Cutting from the right, the default, would throw the question away on long RACE passages.
"""
import numpy as np
import torch

import clm

EMBEDDERS = {                       # key -> (Hugging Face name, label shown in the results table)
    "bge": ("BAAI/bge-small-en-v1.5", "bge-small-en-v1.5 (as is)"),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2 (as is)"),
}
EMBED_MAX_TOKENS = 512              # both embedders have 512 positions

_loaded = {}                        # models are loaded once and reused by every benchmark


# ====================================== QWEN LOG-LIKELIHOOD ======================================
def get_lm():
    """Qwen WITH its text-writing layer (clm.load_backbone() leaves that layer out). Loaded once, frozen."""
    if "lm" not in _loaded:
        from transformers import AutoModelForCausalLM
        device = clm.get_device()
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"Loading {clm.MODEL_NAME} with its output layer on {device} ({dtype}) for the log-likelihood baseline ...")
        lm = AutoModelForCausalLM.from_pretrained(clm.MODEL_NAME, torch_dtype=dtype).to(device).eval()
        for p in lm.parameters():
            p.requires_grad_(False)
        _loaded["lm"] = lm
    return _loaded["lm"]


def encode_case(tok, state_text, options):
    """Token ids for one question: the situation (same recipe as the CLM) and one id list per option."""
    ctx_ids = clm.state_to_ids(tok, [{"role": "user", "content": state_text}])     # ends with "\n\nAnswer:"
    cont_ids = []
    for option in options:
        ids = tok(" " + str(option), add_special_tokens=False)["input_ids"][:clm.MAX_ACTION_TOKENS]   # " Tokyo": a space, then the answer
        cont_ids.append(ids or [tok.pad_token_id])
    return ctx_ids, cont_ids


@torch.inference_mode()
def option_loglik(lm, ctx_ids, cont_ids, pad_id, batched=True):
    """Total log-probability of every option, given the situation. Returns one float per option.

    batched=False: one forward pass per option (the plain method).
    batched=True:  all options in ONE batch. Same answers (up to rounding), a bit faster.
    Sequences are padded on the right, so padding can never change the numbers of the real tokens."""
    device = next(lm.parameters()).device
    n_ctx = len(ctx_ids)
    seqs = [ctx_ids + c for c in cont_ids]
    batches = [list(range(len(seqs)))] if batched else [[i] for i in range(len(seqs))]
    totals = [None] * len(seqs)
    for batch in batches:
        longest = max(len(seqs[i]) for i in batch)
        ids = torch.full((len(batch), longest), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), longest), dtype=torch.long)
        for row, i in enumerate(batch):
            ids[row, :len(seqs[i])] = torch.tensor(seqs[i])
            mask[row, :len(seqs[i])] = 1
        hidden = lm.model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False).last_hidden_state
        for row, i in enumerate(batch):
            n_cont = len(cont_ids[i])
            # The hidden state at position t predicts token t+1, so the states that predict the option tokens are
            # the ones just BEFORE each of them. Only these few positions go through the big output layer (saves memory).
            h = hidden[row, n_ctx - 1:n_ctx + n_cont - 1]
            logp = torch.log_softmax(lm.lm_head(h).float(), dim=-1)
            target = torch.tensor(cont_ids[i], device=device)
            totals[i] = logp.gather(1, target[:, None]).sum()
    return torch.stack(totals).tolist()


def predict_loglik(cases, tok):
    """{row label: [index of the option it picks, per case]} for the log-likelihood baseline (acc and acc_norm)."""
    lm, pad_id = get_lm(), tok.pad_token_id
    picked, picked_norm = [], []
    print(f"  Qwen log-likelihood: scoring {len(cases)} questions (this is the slow baseline) ...")
    for n, case in enumerate(cases, 1):
        ctx_ids, cont_ids = encode_case(tok, case["state"], case["options"])
        totals = np.array(option_loglik(lm, ctx_ids, cont_ids, pad_id, batched=True))
        lengths = np.array([max(len(str(o)), 1) for o in case["options"]], dtype=float)
        picked.append(int(totals.argmax()))
        picked_norm.append(int((totals / lengths).argmax()))
        if n % 250 == 0:
            print(f"    {n}/{len(cases)}")
    return {"Qwen log-likelihood (acc)": picked, "Qwen log-likelihood (acc_norm)": picked_norm}


# ===================================== OFF-THE-SHELF EMBEDDERS =====================================
def get_embedder(key):
    if key not in _loaded:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise SystemExit("The embedder baselines need sentence-transformers. Install it with: pip install sentence-transformers\n"
                             "(or run benchmark.py with --baselines ll / --baselines none to skip them).")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(EMBEDDERS[key][0], device=device)
        model.max_seq_length = EMBED_MAX_TOKENS
        model.tokenizer.truncation_side = "left"          # keep the END of a long situation (the question lives there)
        _loaded[key] = model
    return _loaded[key]


@torch.inference_mode()
def predict_embedder(cases, key):
    """{row label: [index of the option it picks, per case]}: highest cosine between situation and option."""
    label = EMBEDDERS[key][1]
    model = get_embedder(key)
    texts, text_index, option_rows = [], {}, []          # every distinct option text is embedded once, as in benchmark.py
    for case in cases:
        rows = []
        for option in case["options"]:
            if option not in text_index:
                text_index[option] = len(texts)
                texts.append(option)
            rows.append(text_index[option])
        option_rows.append(rows)
    print(f"  {label}: embedding {len(cases)} situations and {len(texts)} distinct options ...")
    zs = model.encode([c["state"] for c in cases], batch_size=16, normalize_embeddings=True,
                      convert_to_tensor=True, show_progress_bar=False)
    za = model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
    return {label: [int((za[rows] @ zs[i]).argmax()) for i, rows in enumerate(option_rows)]}


# ========================================== ENTRY POINTS ==========================================
def parse(choice):
    """The --baselines flag of benchmark.py -> the list of baselines to run."""
    return {"none": [], "ll": ["ll"], "embedders": ["bge", "minilm"], "all": ["ll", "bge", "minilm"]}[choice]


def predict(cases, tok, names):
    """Run the chosen baselines on the cases. Returns {row label: [picked option index per case]}."""
    out = {}
    if "ll" in names:
        out.update(predict_loglik(cases, tok))
    for key in EMBEDDERS:
        if key in names:
            out.update(predict_embedder(cases, key))
    return out
