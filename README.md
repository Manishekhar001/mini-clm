# Mini CLM

A small, from-scratch implementation of a **Contrastive Language Model** (CLM): a "System One" model that takes a *situation* and a list of *options* and returns a probability for every option in a single forward pass, without generating text.

It is a frozen `Qwen3-0.6B` backbone with two small trained heads (a state head and an action head). Both texts are embedded by the backbone, adjusted by their head, and scored by cosine similarity. Only the heads are trained, so the whole pipeline runs on a 6 GB laptop GPU.

> Independent re-implementation of the idea from *Contrastive Language Models* (Kwok et al., Stanford / NVIDIA, Sep 2026). It is not the authors' model and does not reproduce their results (they use an 8B backbone, about 60M training pairs and an H100).

```
situation --> frozen Qwen3-0.6B --> vector --> state head  --+
                                                              +--> cosine --> softmax --> probability per option
option    --> frozen Qwen3-0.6B --> vector --> action head --+
```

```python
import clm
scorer = clm.Scorer()
scorer.rank("Question: What is the capital of Japan?", ["Tokyo", "Osaka", "Paris"])
# {"pre": [...], "mid": [...], "post": [...]}   one probability per option, for each training stage
```

## Results

Zero-shot accuracy on public multiple-choice benchmarks the model never trained on (full test or validation sets, 95% margin of error in the last column).

| | RACE | BoolQ | ARC-Easy | ARC-Challenge | CommonsenseQA |
|---|---|---|---|---|---|
| chance | 25.0% | 50.0% | 25.0% | 25.0% | 20.0% |
| always the longest option | 30.6% | 62.2% | 26.6% | 27.2% | 21.9% |
| frozen Qwen vectors (no heads) | 28.6% | 49.1% | 37.5% | 22.1% | 26.8% |
| **after stage 1 (pre)** | 31.1% | 46.6% | 43.4% | 26.0% | 30.0% |
| **after stage 2 (mid)** | 30.7% | 47.2% | 43.5% | 25.7% | 29.9% |
| **after stage 3 (post)** | 30.5% | 45.8% | 44.0% | 25.9% | 31.5% |
| questions / margin | 4,934 / ±1.3 | 3,270 / ±1.7 | 2,376 / ±2.0 | 1,172 / ±2.5 | 1,221 / ±2.6 |

How to read it:

- **It helps where meaning overlaps.** On ARC-Easy the trained heads add about 6 points over the frozen backbone (37.5% to 44.0%), and on CommonsenseQA about 5 points (26.8% to 31.5%).
- **It does not reason.** ARC-Challenge is at chance. On RACE it matches the "longest option" trick, and on BoolQ it is below the majority-answer baseline (62.2%). Yes/no questions have no option meaning to match.
- **The three stages are indistinguishable on public data.** Differences between them are inside the margin of error.

## Speed

One question at a time (batch size 1) on an RTX 3050 6 GB laptop GPU, median / 95th percentile in milliseconds over 200 questions with 4 options each (`python bench_latency.py`). The baseline is the same Qwen3-0.6B scoring each option by log-likelihood with all four options in one batch, the fair version of that method. The 4-pass version is shown for reference only.

| method | RACE (long passages) | ARC-Easy (short questions) |
|---|---|---|
| CLM | - | - |
| CLM, option vectors cached (only when the options are known in advance) | - | - |
| **Qwen log-likelihood, 1 batch of 4 (baseline)** | - | - |
| Qwen log-likelihood, 4 passes (reference) | - | - |

The CLM saves time because it reads the situation once instead of once per option, so the gain is largest on long passages and small on short questions. Accuracy against the same baseline is in the accuracy table above.

## How it was trained

| stage | data | what it teaches | learning rate |
|---|---|---|---|
| 1. pre | Q&A pairs, 3 hard negatives each | broad question-to-answer matching | 3e-4 |
| 2. mid | the same pairs, up to 10 hard negatives each | fine discrimination between plausible answers | 1e-4 |
| 3. post | agent steps (task + history -> next action), mixed with Q&A replay | decisions in agent settings without forgetting stage 1 | 5e-5 |

Loss: bidirectional InfoNCE with in-batch and mined negatives. Hard negatives come from nearest neighbors in embedding space plus wrong answers written by a small model (`2b_write_negatives.py`).

**Data** (built by `1a` and `1b`, all public): Nemotron-CC Diverse-QA sample and SQuAD 1.1 for Q&A (SQuAD is CC BY-SA 4.0), and `neulab/adp-v2` agent recordings. Check each license before redistributing anything derived from them.

## Lessons from building it

1. **Last-token embeddings of a chat template collapse.** The first version embedded the last token of a ChatML text, which is the newline after `<|im_end|>` and only means "a new turn starts". Every question got almost the same vector (mean pairwise cosine 0.986), so the model could not tell questions apart. Fix: plain text ending in `Answer:`, mean pooling over all tokens, and subtracting the training-set mean vector (centering). Mean cosine fell to about 0.
2. **An easy metric hid it.** The first test picked among 15 random answers from other documents, which only needs "a name vs a date". It reported 79% while same-type choices were at chance. The model is now judged on hard, same-type negatives, and the training log prints a collapse check every epoch.
3. **A deep head made the model worse than raw Qwen.** A three-layer head rebuilt the embedding space and squeezed all answers of one kind together. The head is now residual with a zero-initialized last layer, so it starts as raw Qwen and learns small corrections, with lower learning rates.

## Limitations

- It compares meaning; it does not do arithmetic or notice that something is missing. A card-game question that needs "18 damage vs 10 armor" gets a confident wrong answer.
- Training data is narrow (Wikipedia/web Q&A and agent steps). Games, policies and scoring tasks are out-of-distribution.
- One situation, one question, two or more options. No multi-question input and no numeric score output.
- Probabilities are not calibrated. Do not use it where a wrong answer is costly.
- The agent split is by recording, not by task, so a task seen in training can reappear in test.
- A CUDA GPU is required.

## Run it

```
pip install -r requirements.txt

python 1a_build_qa_pairs.py          # Q&A pairs (stages 1-2)
python 1b_build_agent_pairs.py       # agent pairs (stage 3)
python 2_embed.py                    # embed everything with the frozen backbone (slow, resumable)
python 2b_write_negatives.py         # optional: a small model writes plausible wrong answers
python 3_mine_negatives.py           # mine hard negatives
python 4_train.py --stage all        # train the heads
python 5_demo.py                     # try it, or type your own situations
python benchmark.py                  # public benchmarks + baselines: Qwen log-likelihood, bge-small, all-MiniLM
python bench_latency.py              # speed at batch size 1: median and p95 per question
python eval_stage3.py                # did stage 3 learn anything? held-out agent steps, top-1 / top-3
python push_to_hub.py --repo YOU/mini-clm --dry-run    # prepare a Hugging Face upload
```

The embedding recipe lives in one place, `clm.py` (settings block). Changing any of it requires rerunning steps 2 to 4; old vectors and checkpoints are rejected automatically. `README_detailed.md` has the long step-by-step guide and troubleshooting notes.

## Files

| file | purpose |
|---|---|
| `clm.py` | settings, embedding recipe, centering, head, `Scorer` |
| `1a_`, `1b_` | build Q&A and agent training pairs |
| `2_embed.py`, `2b_`, `3_` | embed, write and mine negatives |
| `4_train.py` | three-stage training and evaluation |
| `5_demo.py`, `demo_cases.json` | interactive demo and examples |
| `benchmark.py` | RACE, BoolQ, ARC, CommonsenseQA evaluation, with the baseline rows next to the CLM |
| `baselines.py` | Qwen3-0.6B log-likelihood scoring and the ready-made embedders (bge-small-en-v1.5, all-MiniLM-L6-v2) |
| `bench_latency.py` | median and p95 latency of the CLM vs Qwen log-likelihood |
| `eval_stage3.py` | stage 3 on held-out agent recordings, against random, a most-common-action prior and `mid` |
| `push_to_hub.py`, `MODEL_CARD.md` | Hugging Face upload and model card template |

## Credit

Idea and recipe: Kwok et al., *Contrastive Language Models*, Stanford / NVIDIA, 2026. Backbone: Qwen3-0.6B.
