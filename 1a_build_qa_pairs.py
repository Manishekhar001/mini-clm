"""Step 1a - build the Q&A pairs for the first two stages (pre-training and mid-training).

Source: NVIDIA Nemotron-CC "Diverse QA". Each row is a web document followed by question/answer
pairs written about it.

    state  = the document + one question about it
    action = the answer to that question

Sources (choose with --source):
  both    (default)  sample + squad together: about 95,000 pairs. Free, public, no login needed.
  sample             nvidia/Nemotron-Pretraining-Dataset-sample, config "Nemotron-CC-Diverse-QA".
                     Small (about 2,000 documents, up to about 8,000 pairs), meant for a laptop.
  squad              rajpurkar/squad (SQuAD 1.1, licence CC BY-SA 4.0): Wikipedia passages, each with several
                     questions and short answers. 87,599 questions. Public, no login needed.
  full               nvidia/Nemotron-CC-v2, config "Diverse-QA". Huge and GATED: NVIDIA must approve your access
                     first and you must log in (`hf auth login`). It is streamed, so only the first
                     --max-pairs pairs are downloaded.

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
PAIRS_PER_DOC = 4           # at most this many pairs from one NVIDIA document (more documents = more variety)
SQUAD_PAIRS_PER_DOC = 8     # SQuAD passages are short, and the typical passage has about 5 questions
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


def nemotron_documents(source, show_rows):
    """NVIDIA rows. Yields (document id, passage, [(question, answer), ...], most pairs to take from it)."""
    if source == "full":
        data = load_dataset("nvidia/Nemotron-CC-v2", "Diverse-QA", split="train", streaming=True)
    else:
        data = load_dataset("nvidia/Nemotron-Pretraining-Dataset-sample", "Nemotron-CC-Diverse-QA", split="train")
    for n, row in enumerate(data, start=1):
        text = row["text"]
        if n <= show_rows:
            print(f"--- raw row {n} (start / end, non-ASCII escaped) ---")
            print(ascii(text[:300]))
            print(ascii(text[-500:]))
        passage, qa = split_row(text)
        doc_id = str(row.get("id") or row.get("uuid") or zlib.crc32(text.encode()))
        yield doc_id, passage, qa, PAIRS_PER_DOC


def squad_documents():
    """SQuAD 1.1. Every row is ONE question, so first group the questions by passage: one passage = one document."""
    data = load_dataset("rajpurkar/squad", split="train")
    by_passage = {}
    for row in data:
        answers = row["answers"]["text"]
        if answers:
            by_passage.setdefault(row["context"], []).append((row["question"].strip(), answers[0].strip()))
    for passage, qa in by_passage.items():
        yield f"squad-{zlib.crc32(passage.encode())}", passage[-MAX_PASSAGE_CHARS:], qa, SQUAD_PAIRS_PER_DOC


def documents(source, show_rows):
    """All documents of the chosen source(s), one after another."""
    if source in ("sample", "full", "both"):
        yield from nemotron_documents("full" if source == "full" else "sample", show_rows)
    if source in ("squad", "both"):
        yield from squad_documents()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["both", "sample", "squad", "full"], default="both")
    parser.add_argument("--max-pairs", type=int, default=100000)
    parser.add_argument("--show-rows", type=int, default=0, help="print the start and end of the first N raw rows (for debugging)")
    args = parser.parse_args()

    rng = random.Random(0)
    pairs_out, rows_seen, rows_with_qa = [], 0, 0
    for doc_id, passage, qa, per_doc in documents(args.source, args.show_rows):
        rows_seen += 1
        if not qa:
            continue
        rows_with_qa += 1
        rng.shuffle(qa)
        split = "val" if zlib.crc32(doc_id.encode()) % 100 < VAL_PERCENT else "train"
        for question, answer in qa[:per_doc]:
            content = (passage + "\n\n" if passage else "") + f"Question: {question}"
            pairs_out.append({"doc": doc_id, "split": split,
                              "state": [{"role": "user", "content": content}],
                              "action": answer[:MAX_ANSWER_CHARS]})
        if len(pairs_out) >= args.max_pairs:
            break

    print(f"Read {rows_seen} documents; {rows_with_qa} of them had questions; built {len(pairs_out)} pairs.")
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
