# Mini CLM: a System One model that runs on a laptop

A **System One** model does not write text. You give it a *situation* and a list of *options*, and it returns a probability for each option in a single pass.

This project builds one small enough to train and run on a **Windows laptop with an RTX 3050 (6 GB)**. It trains in **three stages** (pre-training, mid-training, post-training) and needs only a few hundred MB of disk.

```
situation --> frozen Qwen3-0.6B --> 1024 numbers --> STATE head  --> 256 numbers --+
                                                                                    +--> cosine = score
option    --> frozen Qwen3-0.6B --> 1024 numbers --> ACTION head --> 256 numbers --+
```

Qwen is never changed. Only the two small heads (about 4.7M parameters in total) are trained. Because the Qwen vectors are computed once and saved, each training stage takes minutes.

## The three stages

| Stage | Data | What the model learns |
|---|---|---|
| **1. Pre-training** | Up to about 8,000 Q&A pairs from NVIDIA's Nemotron "Diverse QA" (the public sample has about 2,000 documents) | Each question scores highest with its own answer. The wrong answers are the other answers in the same batch. |
| **2. Mid-training** | The same Q&A pairs plus up to 10 **hard negatives** per question | To tell the right answer from wrong answers that look close to it. The hard negatives are real answers from other documents, mined in step 3. |
| **3. Post-training** | Up to 300 agent decisions from each text-only config of Agent Data Protocol v2, mixed 60% agent / 40% Q&A replay | To choose good next actions for an AI agent, without forgetting the Q&A skills. The temperature is frozen. |

Each stage starts from the heads the previous stage saved. Learning rates: stage 1 uses `2e-3 × sqrt(1024/width) × sqrt(batch/1024)`, stage 2 uses 1e-3, stage 3 uses 3e-4.

**Why Qwen3-0.6B and not Qwen3.5?** Qwen3.5 starts at 0.8B and has a newer architecture that I have not tested with this code. Qwen3-0.6B is small, fast and well supported. You can try another model by changing `MODEL_NAME` in `clm.py`.

## Files

```
clm.py                   settings, frozen backbone, embedding function, Head network, save/load, Scorer (edit settings here)
1a_build_qa_pairs.py     NVIDIA Diverse QA text -> (document + question, answer) pairs     -> data/qa_pairs.jsonl
1b_build_agent_pairs.py  agent recordings -> (conversation so far, next action)            -> data/agent_pairs.jsonl
2_embed.py               Qwen turns every text into a vector (the slow step)               -> data/embeddings_*.npz
3_mine_negatives.py      find hard negatives for the Q&A pairs                             -> data/negatives.npz
4_train.py               train stage 1, 2, 3, then print one results table                 -> data/heads_pre/mid/post.pt
5_demo.py                inference: give it a situation + options, see each stage's probabilities
requirements.txt
```

Read them in order. Each file starts with a plain-English summary of what it does.

## Setup (Windows)

You need [uv](https://docs.astral.sh/uv/) (install it with `pip install uv` if you do not have it) and an NVIDIA driver. uv downloads Python 3.12 by itself if it is missing. Open PowerShell in this folder:

```powershell
uv venv --python 3.12                  # creates the .venv folder
.venv\Scripts\activate                  # the prompt now starts with (.venv) or the folder name
uv pip install -r requirements.txt     # installs everything, including PyTorch with CUDA
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

With the environment activated, plain `python` uses it. If you do not activate it, write `uv run --no-sync python ...` instead of `python ...`. Keep the `--no-sync`: in a folder that has a `pyproject.toml`, a plain `uv run` first re-syncs the environment and can put the CPU-only PyTorch back.

`requirements.txt` already points at PyTorch's CUDA 12.6 builds, so one command installs everything. The last line **must print a version ending in `+cu126`, then `True`, then your GPU name**. If it prints `False`, see Troubleshooting. If your driver is too old for CUDA 12.6 (`nvidia-smi` shows a lower "CUDA Version"), update the driver or change `cu126` to `cu118` in `requirements.txt`.

If PowerShell refuses to activate the environment, run `Set-ExecutionPolicy -Scope Process Bypass` and try again.

## Run

Run everything from this folder with the environment activated (`.venv\Scripts\activate`). Without activating, put `uv run --no-sync` in front, for example `uv run --no-sync python 2_embed.py`.

### A. Quick test of the whole pipeline (about 10 to 30 minutes)

Do this first. This code was written without being run, so a small bug is better found now than after a multi-hour step 2.

```powershell
python 1a_build_qa_pairs.py --max-pairs 2000      # NVIDIA's small public sample, 2,000 pairs
python 1b_build_agent_pairs.py --per-config 100   # a small slice of the agent data
python 2_embed.py                                 # Qwen reads every text once (GPU)
python 3_mine_negatives.py                        # seconds
python 4_train.py --stage all --epochs 3          # three short stages, then the results table
python 5_demo.py                                  # inference with the built-in examples
```

If step 4 stops with "Too little data", raise the numbers (for example `--max-pairs 4000` and `--per-config 200`) and rerun from step 2.

### B. The real run (several hours, mostly step 2)

```powershell
hf auth login                                                     # once. First accept NVIDIA's terms on the dataset page on Hugging Face
python 1a_build_qa_pairs.py --source full --max-pairs 100000      # about 25,000 documents, streamed
python 1b_build_agent_pairs.py --per-config 1000
python 2_embed.py                                                 # the slow step; resumes if stopped
python 3_mine_negatives.py
python 4_train.py --stage all                                     # add --batch 4096 for more wrong answers per step
python 5_demo.py
```

Running 1a or 1b again deletes the vectors and negatives built from the old pairs, so rerun steps 2, 3 and 4 after it.

### Every file on its own

| Step | Command | Options | Writes |
|---|---|---|---|
| 1a | `python 1a_build_qa_pairs.py` | `--source sample` (default) or `full`; `--max-pairs N`; `--show-rows N` (print N raw rows, for debugging the format) | `data/qa_pairs.jsonl` |
| 1b | `python 1b_build_agent_pairs.py` | `--per-config N` (pairs taken from each dataset config, default 300) | `data/agent_pairs.jsonl` |
| 2 | `python 2_embed.py` | `--only qa` or `--only agent` (just one dataset) | `data/embeddings_qa.npz`, `data/embeddings_agent.npz` |
| 3 | `python 3_mine_negatives.py` | none | `data/negatives.npz` |
| 4 | `python 4_train.py --stage all` | `--stage pre`, `mid`, `post` or `all`; `--batch N`; `--lr X`; `--epochs N` | `data/heads_pre.pt`, `heads_mid.pt`, `heads_post.pt` |
| 5 | `python 5_demo.py` | `--stage pre/mid/post/all`; `--state "..." --option "..." --option "..."`; `--interactive`; `--file cases.json` | prints a table |

Steps 4 stage by stage: `python 4_train.py --stage pre`, then `--stage mid`, then `--stage post`. Each stage needs the one before it.

`2_embed.py` saves its progress every 1000 pairs. If you stop it or the laptop sleeps, run it again and it carries on. You can also run `4_train.py --stage pre`, then `mid`, then `post`, one at a time.

**Expected time and memory.** These are estimates from the model size, not measurements from your machine. Step 2 embeds two texts (a state and an action) for every pair, so a few tens of thousands of texts in total. With the default sample (about 8,000 pairs) I would guess tens of minutes to an hour or two on an RTX 3050, using a few GB of GPU memory. With `--source full --max-pairs 100000` and 1024-token situations, expect several hours. I have not measured either. The script prints its own speed and time remaining after the first chunk of 1,000 pairs, so you will know quickly. Step 4 should take a few minutes in total.

## Using the model (inference)

Once `4_train.py` has finished, the three trained stages are saved as `data/heads_pre.pt`, `heads_mid.pt` and `heads_post.pt`. `5_demo.py` loads Qwen and every saved stage, scores your options once, and prints one table. Each column is a training stage and a `*` marks the option that stage picks.

```powershell
python 5_demo.py                                   # built-in Q&A and agent examples
python 5_demo.py --state "Question: What is the capital of Japan?" --option "Tokyo" --option "Osaka" --option "Paris"
python 5_demo.py --interactive                     # type situations and options, one after another
python 5_demo.py --file my_cases.json              # a list of {"state": "...", "options": ["...", "..."]}
python 5_demo.py --stage post                      # just one stage (pre, mid or post)
```

From your own Python code (run it from this folder):

```python
import clm
scorer = clm.Scorer()                              # loads Qwen and the saved heads
scorer.rank("Question: Capital of Japan?", ["Tokyo", "Paris"])
# {'pre': [0.8, 0.2], 'mid': [0.9, 0.1], 'post': [0.9, 0.1]}   (example shape, not real output)
```

It is fast because there is only one pass through Qwen per text and no text is generated. Qwen loads once when you start, and after that each case takes a fraction of a second on a GPU.

Tip: stages 1 and 2 only saw Q&A written like `Question: ...` with short plain-text answers. Stage 3 also saw agent decisions written like `tool_name({json arguments})`, so options in that style work best for agent situations.

## Reading the results

`4_train.py` ends with a table like this (the numbers here are placeholders, not results):

```
                      Q&A (1 of 16)   Tricky Q&A   Agent (1 of 16)
random guessing               6.2%        xx.x%             6.2%
raw Qwen vectors              ...
after 'pre'                   ...
after 'mid'                   ...
after 'post'                  ...
```

- **Q&A:** the right answer has to beat 15 random answers from other documents.
- **Tricky Q&A:** the right answer has to beat its own mined hard negatives. This should improve most after stage 2.
- **Agent:** the right agent action has to beat 15 random actions from other recordings. This should improve most after stage 3.
- **All three tests use pairs the model never trained on**, and the split is by document or recording, so the model never sees the same document in training and testing.
- The best epoch of each stage is chosen on the same test pairs, so the numbers are slightly optimistic.

What to look for: stage 2 should lift "Tricky Q&A", and stage 3 should lift "Agent" while "Q&A" and "Tricky" stay close to where they were. If Q&A or Tricky fall a lot after stage 3, the Q&A replay is not doing its job.

## Settings you may want to change

| Setting | Where | What it does |
|---|---|---|
| `MODEL_NAME` | `clm.py` | The backbone. Rerun steps 2, 3 and 4 after changing it. |
| `MAX_STATE_TOKENS`, `MAX_ACTION_TOKENS` | `clm.py` | How much of a situation / option Qwen reads (1024 and 128). Lower = faster and less memory. |
| `EMBED_BATCH` | `clm.py` | Lower it (try 4 or 2) if step 2 runs out of GPU memory. |
| `HEAD_WIDTH`, `PROJ_DIM` | `clm.py` | Head size. |
| `--max-pairs` | step 1a | Q&A pairs to build. More data = better heads, but a slower step 2. The sample only has about 2,000 documents, so it tops out around 8,000 pairs. |
| `--source full` | step 1a | Use the full NVIDIA dataset (see below) for more Q&A data. |
| `--per-config` | step 1b | Agent pairs taken from each dataset config. |
| `STAGES` table | `4_train.py` | Learning rate, epochs, and patience of each stage. |
| `--batch`, `--lr`, `--epochs` | step 4 | Batch size and overrides for every stage. |

## More Q&A data (optional)

`--source full` reads `nvidia/Nemotron-CC-v2` (Diverse-QA), which is gated: open its page on Hugging Face, accept NVIDIA's terms (they limit use to model training), and run `hf auth login` first. It is streamed, so only the first `--max-pairs` pairs are downloaded. I could not check this path myself because the dataset is gated.

## How it works

1. **Pairs.** Q&A: a state is a document plus one question about it, and the action is the answer. Agent: a state is the conversation so far, and the action is what the agent did next.
2. **Vectors.** Qwen reads a text and we keep the hidden state of its **last token**, which has seen the whole text. We keep the end of long situations and the start of long options.
3. **Heads.** Each head turns a Qwen vector into a shorter one. The score is the cosine of the two vectors times a learned scale (the temperature).
4. **Contrastive training.** In a batch, every state must score highest with its own action and lower with everyone else's. Pairs from the same document or recording are not counted as wrong answers, because their actions can look alike.
5. **Hard negatives.** In stages 2 and 3, each Q&A pair also gets up to 10 extra wrong answers that look close to the right one.
6. **Using it.** `5_demo.py` embeds your situation and options, scores each option, and applies softmax to get probabilities.

## Troubleshooting

- **Step 1a says "Too few pairs".** The script guesses the Q&A format (`Question:` / `Answer:` markers) because I could not see raw rows when writing it. Run `python 1a_build_qa_pairs.py --show-rows 3`, look at the raw text it prints, and adjust the `PAIR` pattern at the top of the file.
- **`torch.cuda.is_available()` is False.** You probably installed the CPU build of PyTorch. Run `uv pip uninstall torch`, then `uv pip install "torch>=2.6,<2.8" --index-url https://download.pytorch.org/whl/cu126` (with plain pip: `pip uninstall torch`, then the same `pip install ...`). A version ending in `+cpu` confirms it was the CPU build. Also check that `nvidia-smi` works in PowerShell.
- **Out of memory in step 2.** Set `EMBED_BATCH = 2` and/or `MAX_STATE_TOKENS = 256` in `clm.py`. Close other GPU-heavy apps such as games and browsers.
- **NaN error in step 2.** Lower `EMBED_BATCH`. Very old drivers can also cause it.
- **Step 1b skips some configs.** That is expected: some dataset folders have a different layout. It prints which ones and carries on.
- **Downloads fail or are slow.** Rerun the same command. Step 2 resumes by itself, and Hugging Face caches the model after the first download (about 1.5 GB).
- **`UnicodeEncodeError` in the console.** Run `chcp 65001` first, or set `PYTHONUTF8=1`.
- **Stage `mid` or `post` says the previous stage is missing.** Each stage continues from the one before it. Run `python 4_train.py --stage all`, or run them in order.

## Limits to be honest about

- **This code has not been run.** It was written without access to a GPU or PyTorch, so expect a small bug or two on the first run.
- It is a small-scale model: a 0.6B backbone and a few thousand pairs per stage. It has not been tested on agent benchmarks, so treat the three simple tests above as a sanity check, not a measure of real-world quality.
- With only a few thousand pairs, expect fast training and some overfitting. The early-stopping setting in `4_train.py` is there for that reason.
- Mined hard negatives are real answers that look similar, not wrong answers written for each question, so they are noisy. Some may even be acceptable answers to the question. Step 3 prints samples so you can judge.
- The Nemotron data is under the NVIDIA Open Data License (the sample) or the NVIDIA Data Agreement for Model Training (the full set, which limits use to model training). Parts of it were generated with Qwen and DeepSeek models, so their licences may apply to models trained on it.

## Background

The idea and the three-stage recipe come from Contrastive Language Models (Kwok et al., Stanford and NVIDIA): a frozen language model plus small trained heads, learned with a contrastive loss on state/action pairs. See [their write-up](https://contrastive-lm.notion.site).
