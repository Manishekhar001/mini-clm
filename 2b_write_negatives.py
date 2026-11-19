"""Step 2b - let a small language model WRITE hard negatives for some of the Q&A pairs (optional).

A hard negative is a wrong answer that looks right. Step 3 already MINES hard negatives: real answers of other
questions that look close to the right answer. Those were not made for the question at hand, and a few of them
could even be correct. Here a local model (Qwen3-1.7B, about 3.5 GB on the GPU) is shown a question and its right
answer and is asked for 3 believable WRONG answers. Step 3 then puts the written ones first and the mined ones after.

Only some pairs get written negatives (--num-pairs), because writing for every pair would take hours.
The writer sees the question and the right answer, not the passage, so each prompt is short and fast.

Honest limits: a small model sometimes writes an answer that is actually correct. Exact copies of the right answer
are removed, but nothing else is checked. Look at the examples that this script and step 3 print.

It saves its work after every batch (data/synthetic_negatives.jsonl). If you stop it, run it again and it carries on.
At the end it turns every written answer into a vector with the same Qwen as step 2 -> data/synthetic_negatives.npz.

Run:
    python 2b_write_negatives.py                      # 15,000 pairs
    python 2b_write_negatives.py --num-pairs 500      # quick test
Then run step 3 again:   python 3_mine_negatives.py
"""
import argparse
import json
import os
import random
import re
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import clm

WRITER = "Qwen/Qwen3-1.7B"    # the model that writes the wrong answers
PER_QUESTION = 3              # wrong answers asked for, per question
MAX_NEW_TOKENS = 120          # the writer may produce this many tokens per question
MAX_ANSWER_IN_PROMPT = 300    # the right answer is cut to this many characters inside the prompt
MAX_NEG_CHARS = 300           # a written wrong answer is cut to this many characters

PROMPT = ("Question: {question}\n"
          "Correct answer: {answer}\n\n"
          "Write {n} different WRONG answers to this question. Each wrong answer must look believable: the same topic, "
          "the same kind of answer, the same style and a similar length as the correct answer, but it must not be correct.\n"
          "Write one wrong answer per line. No numbering, no bullets, no explanations.")

LABEL = re.compile(r"^\s*(?:[-*•]|\d+[.)]|wrong answer\s*\d*\s*[:.\-])\s*", re.I)   # "1.", "- ", "Wrong answer 2:" ...


def normalise(text):
    """Lower case, one space between words, no end dots: so that 'Paris.' and 'paris' count as the same answer."""
    return " ".join(text.lower().split()).strip(" .")


def clean(lines, answer, n):
    """The writer's lines -> up to n usable wrong answers: labels removed, no copy of the right answer, no repeats."""
    seen = {normalise(answer)}
    out = []
    for line in lines:
        line = LABEL.sub("", line).strip().strip('"').strip()
        if not line or line.endswith(":"):           # empty, or an intro line like "Here are 3 wrong answers:"
            continue
        key = normalise(line)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(line[:MAX_NEG_CHARS])
        if len(out) == n:
            break
    return out


def load_written(pairs):
    """Records written in earlier runs, keeping only the ones that still belong to the current pairs."""
    done = {}
    if os.path.exists(clm.SYN_TEXT_PATH):
        with open(clm.SYN_TEXT_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue                          # a half-written last line
                i = rec.get("pair", -1)
                if 0 <= i < len(pairs) and rec.get("k") == clm.pair_key(pairs[i]):
                    done[i] = rec["negatives"]
    return done


def write_negatives(pairs, todo, batch_size):
    """Ask the writer model for wrong answers, batch by batch. Every batch is appended to the .jsonl file."""
    device = clm.get_device()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading {WRITER} on {device} ({dtype}) ...")
    tok = AutoTokenizer.from_pretrained(WRITER)
    tok.padding_side = "left"                         # generation continues from the END of each prompt
    model = AutoModelForCausalLM.from_pretrained(WRITER, torch_dtype=dtype).to(device).eval()
    torch.manual_seed(0)

    prompts = {}
    for i in todo:
        user = PROMPT.format(n=PER_QUESTION, question=clm.question_of(pairs[i]), answer=pairs[i]["action"][:MAX_ANSWER_IN_PROMPT])
        prompts[i] = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    todo = sorted(todo, key=lambda i: len(prompts[i]))   # similar lengths together = less padding

    t0, done_now = time.time(), 0
    with open(clm.SYN_TEXT_PATH, "a", encoding="utf-8") as out:
        for b, start in enumerate(range(0, len(todo), batch_size)):
            batch = todo[start:start + batch_size]
            enc = tok([prompts[i] for i in batch], return_tensors="pt", padding=True).to(device)
            with torch.inference_mode():
                gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.7,
                                     top_p=0.8, top_k=20, pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for i, text in zip(batch, texts):
                negatives = clean(text.splitlines(), pairs[i]["action"], PER_QUESTION)
                out.write(json.dumps({"pair": i, "k": clm.pair_key(pairs[i]), "negatives": negatives}, ensure_ascii=False) + "\n")
            out.flush()                               # only whole batches count as done
            done_now += len(batch)
            if b % 10 == 0 or start + batch_size >= len(todo):
                rate = done_now / (time.time() - t0)
                left_min = (len(todo) - done_now) / rate / 60
                print(f"  {done_now}/{len(todo)} pairs written  ({rate:.1f} pairs/s, about {left_min:.0f} min left)", flush=True)
    del model
    torch.cuda.empty_cache()                          # give the GPU back before the next Qwen is loaded


def save_vectors(pairs, done):
    """Embed every written wrong answer with the SAME backbone as step 2, and save them next to their pair numbers."""
    idx, texts = [], []
    for i in sorted(done):
        for t in done[i]:
            idx.append(i)
            texts.append(t)
    if not texts:
        raise SystemExit("The writer produced no usable wrong answers. Try --num-pairs 500 and look at the output.")
    tok, model = clm.load_backbone()
    vectors = clm.embed(model, tok, [clm.action_to_ids(tok, t) for t in texts])
    np.savez(clm.SYN_NEG_PATH, pair_idx=np.array(idx, dtype=np.int64),
             keys=np.array([clm.pair_key(pairs[i]) for i in idx], dtype=np.int64),
             texts=np.array(texts), vectors=vectors, model_name=np.array(clm.MODEL_NAME))
    n_none = sum(1 for i in done if not done[i])
    print(f"Saved {len(texts)} written wrong answers for {len(done) - n_none} pairs ({n_none} pairs got none) to {clm.SYN_NEG_PATH}")

    def safe(text, n=110):
        return " ".join(text.split())[:n].encode("ascii", "replace").decode()
    print("\nA few examples to check by eye:")
    for i in random.Random(1).sample([i for i in done if done[i]], min(4, len(done) - n_none)):
        print("\n  Question:       ", safe(clm.question_of(pairs[i])))
        print("  Correct answer: ", safe(pairs[i]["action"]))
        for t in done[i]:
            print("  Wrong (written):", safe(t))
    print("\nNext: python 3_mine_negatives.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-pairs", type=int, default=15000, help="how many Q&A pairs get written negatives")
    parser.add_argument("--batch", type=int, default=16, help="questions per GPU batch (lower it if you run out of memory)")
    args = parser.parse_args()

    if not os.path.exists(clm.QA_PAIRS_PATH):
        raise SystemExit(f"{clm.QA_PAIRS_PATH} not found. Run 1a_build_qa_pairs.py first.")
    with open(clm.QA_PAIRS_PATH, encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f]

    order = list(range(len(pairs)))
    random.Random(0).shuffle(order)                   # the same random pairs every run, so a rerun carries on
    chosen = order[:args.num_pairs]
    done = load_written(pairs)
    todo = [i for i in chosen if i not in done]
    print(f"{len(chosen)} pairs chosen, {len(done)} already written, {len(todo)} still to write.")
    if todo:
        write_negatives(pairs, todo, args.batch)
        done = load_written(pairs)
    save_vectors(pairs, done)


if __name__ == "__main__":
    main()
