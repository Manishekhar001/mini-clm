"""Step 1a - build the Q&A pairs for the first two stages (pre-training and mid-training).

Source: NVIDIA Nemotron-CC "Diverse QA". Each row is a web document followed by question/answer
pairs written about it.

    state  = the document + one question about it
    action = the answer to that question

Two sources (choose with --source):
  sample  (default)  nvidia/Nemotron-Pretraining-Dataset-sample, config "Nemotron-CC-Diverse-QA".
                     Small (about 2,000 documents), meant for a laptop.
  full               nvidia/Nemotron-CC-v2, config "Diverse-QA". Huge and gated: accept NVIDIA's terms on its
                     Hugging Face page and log in first (`hf auth login`). It is streamed, so only the
                     first --max-pairs pairs are downloaded.

IMPORTANT: I could not see a raw row of this data when writing this script. It assumes the Q&A pairs are
written with "Question:" and "Answer:" markers after the document text. Step 1a prints what it finds. If it
stops with an error, run it with --show-rows 3, look at the raw text, and adjust PAIR below.

Run:
    python 1a_build_qa_pairs.py
    python 1a_build_qa_pairs.py --max-pairs 4000      # smaller and faster
"""
import argparse
import glob
import json
import os
import random
import re
import shutil
import zlib

from datasets import load_dataset

# One "Question: ... Answer: ..." pair. Works for "- Question: q Answer: a" bullets and for separate lines.
PAIR = re.compile(r"Question:\s*(.+?)\s*Answer:\s*(.+?)(?=\s*(?:-\s*)?Question:|\Z)", re.S)
PAIRS_PER_DOC = 4           # at most this many pairs from one document (more documents = more variety)
MAX_PASSAGE_CHARS = 5000    # keep the last 5000 characters of the document (we cut to 1024 tokens later anyway)
MAX_ANSWER_CHARS = 1000
VAL_PERCENT = 10            # about 10% of DOCUMENTS are held out for testing (never trained on)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")   # same folder as clm.py uses
QA_PAIRS_PATH = os.path.join(DATA_DIR, "qa_pairs.jsonl")


def split_row(text):
    """One row of text -> (document text, [(question, answer), ...])."""
    matches = list(PAIR.finditer(text))
    if not matches:
        return "", []
    passage = text[:matches[0].start()].rstrip(" \n-")
    pairs = [(m.group(1).strip(), m.group(2).strip()) for m in matches]
    return passage[-MAX_PASSAGE_CHARS:], pairs


def open_source(source):
    if source == "sample":
        return load_dataset("nvidia/Nemotron-Pretraining-Dataset-sample", "Nemotron-CC-Diverse-QA", split="train")
    return load_dataset("nvidia/Nemotron-CC-v2", "Diverse-QA", split="train", streaming=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["sample", "full"], default="sample")
    parser.add_argument("--max-pairs", type=int, default=100000)
    parser.add_argument("--show-rows", type=int, default=0, help="print the start and end of the first N raw rows (for debugging)")
    args = parser.parse_args()

    rng = random.Random(0)
    pairs_out, rows_seen, rows_with_qa = [], 0, 0
    for row in open_source(args.source):
        rows_seen += 1
        text = row["text"]
        if rows_seen <= args.show_rows:
            print(f"--- raw row {rows_seen} (start / end, non-ASCII escaped) ---")
            print(ascii(text[:300]))
            print(ascii(text[-500:]))
        passage, qa = split_row(text)
        if not qa:
            continue
        rows_with_qa += 1
        rng.shuffle(qa)
        doc_id = str(row.get("id") or row.get("uuid") or zlib.crc32(text.encode()))
        split = "val" if zlib.crc32(doc_id.encode()) % 100 < VAL_PERCENT else "train"
        for question, answer in qa[:PAIRS_PER_DOC]:
            content = (passage + "\n\n" if passage else "") + f"Question: {question}"
            pairs_out.append({"doc": doc_id, "split": split,
                              "state": [{"role": "user", "content": content}],
                              "action": answer[:MAX_ANSWER_CHARS]})
        if len(pairs_out) >= args.max_pairs:
            break

    print(f"Read {rows_seen} rows; {rows_with_qa} of them contained Question:/Answer: pairs; built {len(pairs_out)} pairs.")
    if len(pairs_out) < 200:
        raise SystemExit("Too few pairs. The data format is probably different from what this script expects.\n"
                         "Run again with --show-rows 3 and look at the raw text, then adjust PAIR at the top of this file.")

    random.Random(0).shuffle(pairs_out)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(QA_PAIRS_PATH, "w", encoding="utf-8") as f:
        for p in pairs_out:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # New pairs make the old Q&A vectors and negatives invalid, so remove them.
    for old in glob.glob(os.path.join(DATA_DIR, "chunks_qa_*")):
        shutil.rmtree(old)
    for stale in (os.path.join(DATA_DIR, "embeddings_qa.npz"), os.path.join(DATA_DIR, "negatives.npz")):
        if os.path.exists(stale):
            os.remove(stale)

    n_val = sum(p["split"] == "val" for p in pairs_out)
    print(f"Saved {len(pairs_out)} pairs ({len(pairs_out) - n_val} train, {n_val} test) to {QA_PAIRS_PATH}")
    print("Next: python 1b_build_agent_pairs.py, then python 2_embed.py")


if __name__ == "__main__":
    main()
