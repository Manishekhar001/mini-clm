---
@@LICENSE_LINE@@
base_model: @@BASE_MODEL@@
tags:
- contrastive-learning
- system-one
- decision-making
- action-scoring
---

# Mini CLM

A small **Contrastive Language Model**: given a *situation* and a list of *options*, it returns a probability for every option in one forward pass, without generating text.

It is a frozen `@@BASE_MODEL@@` backbone plus two small trained heads (a state head and an action head). Situation and options are embedded by the backbone, adjusted by their head, and scored by cosine similarity. Only the heads are trained, on a 6 GB laptop GPU.

This is an independent small-scale re-implementation of the idea in *Contrastive Language Models* (Kwok et al., Stanford / NVIDIA, 2026). It is not the authors' model and does not reproduce their results.

**This is not a `transformers` model.** Use it through the included `clm.py`. A CUDA GPU is required.

## Usage

```python
import sys
from huggingface_hub import snapshot_download

path = snapshot_download("@@REPO@@")      # add token="..." if the repo is private
sys.path.insert(0, path)
import clm

scorer = clm.Scorer()                      # loads the backbone and the trained heads
print(scorer.rank("Question: What is the capital of Japan?", ["Tokyo", "Osaka", "Paris"]))
# {"pre": [...], "mid": [...], "post": [...]}   one probability per option, for each training stage
```

`scorer.rank(situation, options)` takes one situation, one question and two or more options. Write situations the way the model was trained: a passage first when there is one, then `Question: ...`. Bare questions with no context score worse. `python 5_demo.py` in the downloaded folder runs a ready-made demo.

## Results

Zero-shot accuracy on public multiple-choice benchmarks the model never trained on (full test or validation sets):

| | RACE | BoolQ | ARC-Easy | ARC-Challenge | CommonsenseQA |
|---|---|---|---|---|---|
| chance | 25.0% | 50.0% | 25.0% | 25.0% | 20.0% |
| always the longest option | 30.6% | 62.2% | 26.6% | 27.2% | 21.9% |
| frozen backbone vectors (no heads) | 28.6% | 49.1% | 37.5% | 22.1% | 26.8% |
| after stage 1 (`pre`) | 31.1% | 46.6% | 43.4% | 26.0% | 30.0% |
| after stage 2 (`mid`) | 30.7% | 47.2% | 43.5% | 25.7% | 29.9% |
| after stage 3 (`post`) | 30.5% | 45.8% | 44.0% | 25.9% | 31.5% |
| questions / 95% margin | 4,934 / ±1.3 | 3,270 / ±1.7 | 2,376 / ±2.0 | 1,172 / ±2.5 | 1,221 / ±2.6 |

- It helps where the answer shares meaning with the question: about +6 points over the frozen backbone on ARC-Easy and about +5 on CommonsenseQA.
- It does not reason. ARC-Challenge is at chance, RACE is no better than picking the longest option, and BoolQ is below the 62.2% majority-answer baseline.
- The three stages are within the margin of error of each other on public data.

Held-out pairs from the training distribution (same sources as training, never trained on):

@@METRICS_TABLE@@

"Q&A (1 of 16)" uses random wrong answers and is easy. "Tricky Q&A" uses plausible wrong answers and is the number to trust.

## Embedding recipe (must match exactly)

@@RECIPE@@

`data/centering.npz` holds the average state and action vectors that are subtracted at inference. Checkpoints trained with a different recipe or head design are refused by `clm.py`.

## Training

| stage | data | learning rate |
|---|---|---|
| 1. `pre` | question/answer pairs with 3 hard negatives each | 3e-4 |
| 2. `mid` | the same pairs with up to 10 hard negatives each | 1e-4 |
| 3. `post` | agent steps (task and history to next action) mixed with Q&A replay | 5e-5 |

Loss: bidirectional InfoNCE with in-batch and mined hard negatives. The heads are residual: the last layer starts at zero, so the model begins as the raw backbone and learns small corrections.

**Data** (all public): Nemotron-CC Diverse-QA sample and SQuAD 1.1 (CC BY-SA 4.0) for question answering, and `neulab/adp-v2` agent recordings for stage 3. Review each dataset's license before reusing this model.

## Limitations

- It compares meaning and does no step-by-step reasoning. Questions that need arithmetic, or that depend on something being absent, can get confident wrong answers.
- The training data is narrow (web and Wikipedia-style Q&A, agent steps). Games, policies and scoring tasks are out of distribution.
- One situation, one question, two or more options. No multi-question input and no numeric score output.
- Probabilities are not calibrated. Do not use it where a wrong answer is costly.

## Credit

Kwok et al., *Contrastive Language Models*, Stanford / NVIDIA, 2026. Backbone: `@@BASE_MODEL@@`.
