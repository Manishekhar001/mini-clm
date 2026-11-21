"""Push the trained model to the Hugging Face Hub.

What gets uploaded (and nothing else - your .env, your training data and the big vector files stay on your laptop):
    clm.py, 5_demo.py, demo_cases.json, benchmark.py, requirements.txt     the code needed to use the model
    data/heads_pre.pt, heads_mid.pt, heads_post.pt                         the trained heads (the model itself, a few MB each)
    data/centering.npz                                                     the average vectors the model needs at inference
    config.json                                                            the recipe, so anyone can see how it was built
    README.md                                                              a model card, written for you from your results

This is NOT a transformers model: it is a small set of heads on top of a frozen Qwen backbone, used through clm.py.
The model card explains how to load it.

First time only: log in with a token that has WRITE access (huggingface.co/settings/tokens):
    hf auth login            (or: huggingface-cli login, or set the HF_TOKEN environment variable)

Run:
    python push_to_hub.py --repo YOUR_USERNAME/mini-clm --dry-run    # prepare everything in ./hf_upload, upload nothing
    python push_to_hub.py --repo YOUR_USERNAME/mini-clm              # upload as a PRIVATE repo (the safe default)
    python push_to_hub.py --repo YOUR_USERNAME/mini-clm --public     # make it public
    python push_to_hub.py --repo YOUR_USERNAME/mini-clm --license apache-2.0   # add a license line to the model card

Check the licenses of your training data before you make the repo public (see the model card notes).
"""
import argparse
import json
import os
import shutil
import sys

import torch

import clm

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE_DIR = os.path.join(HERE, "hf_upload")          # the folder that gets uploaded (made fresh every run)
CODE_FILES = ["clm.py", "5_demo.py", "demo_cases.json", "benchmark.py", "requirements.txt"]
STAGES = ("pre", "mid", "post")


def load_checkpoint_info(stage):
    """Read a saved stage's settings and test numbers. Stops if it does not match the current clm.py."""
    path = clm.ckpt_path(stage)
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("recipe") != clm.RECIPE_ID or ck.get("arch") != clm.HEAD_ARCH:
        raise SystemExit(f"{path} was trained with a different recipe or head design than clm.py says now. "
                         "Retrain (python 4_train.py --stage all) before pushing, or the upload would not work.")
    return ck


def percent(x):
    return f"{x:.1%}" if isinstance(x, (int, float)) else "n/a"


def model_card(repo, infos, license_id):
    """The README.md shown on the model page: MODEL_CARD.md with its @@TOKENS@@ filled in from your run."""
    template_path = os.path.join(HERE, "MODEL_CARD.md")
    if not os.path.exists(template_path):
        return builtin_model_card(repo, infos, license_id)
    with open(template_path, encoding="utf-8") as f:
        text = f.read()
    recipe = (f"- Backbone: `{clm.MODEL_NAME}` (frozen)\n"
              f"- Situation text: `{clm.STATE_MODE}` mode, suffix `{clm.STATE_SUFFIX!r}`\n"
              f"- Option text prefix: `{clm.ACTION_PREFIX!r}`\n"
              f"- Pooling: `{clm.POOLING}`; centering: `{clm.CENTER}`\n"
              f"- Head: residual MLP (`{clm.HEAD_ARCH}`), width {clm.HEAD_WIDTH}\n"
              f"- Recipe id: `{clm.RECIPE_ID}`")
    table = ["| stage | Q&A (1 of 16) | Tricky Q&A | Agent (1 of 16) |", "|---|---|---|---|"]
    for stage, ck in infos.items():
        m = ck.get("metrics", {})
        table.append(f"| {stage} | {percent(m.get('qa_acc'))} | {percent(m.get('tricky_acc'))} | {percent(m.get('agent_acc'))} |")
    fills = {"@@LICENSE_LINE@@": f"license: {license_id}" if license_id else "",
             "@@BASE_MODEL@@": clm.MODEL_NAME, "@@REPO@@": repo, "@@RECIPE@@": recipe, "@@METRICS_TABLE@@": "\n".join(table)}
    for token, value in fills.items():
        text = text.replace(token, value)
    return text


def builtin_model_card(repo, infos, license_id):
    """The older, built-in model card (used only when MODEL_CARD.md is missing)."""
    lines = []
    lines.append("---")
    if license_id:
        lines.append(f"license: {license_id}")
    lines.append(f"base_model: {clm.MODEL_NAME}")
    lines.append("tags:\n- contrastive-learning\n- system-one\n- decision-making\n- action-scoring")
    lines.append("---\n")
    lines.append("# Mini CLM (a small Contrastive Language Model)\n")
    lines.append("A small, from-scratch version of the *Contrastive Language Model* idea: instead of writing text, the model "
                 "takes a **situation** and a list of **options**, and returns a probability for every option in one pass.\n")
    lines.append("It is **a frozen `" + clm.MODEL_NAME + "` backbone plus two small trained heads** (a state head and an action head). "
                 "Both texts are embedded by the frozen backbone, adjusted by their head, and scored with cosine similarity. "
                 "Only the heads are trained. This is not a `transformers` model: use it through the included `clm.py`.\n")

    lines.append("## How to use\n")
    lines.append("```python\nimport sys\nfrom huggingface_hub import snapshot_download\n\n"
                 f"path = snapshot_download(\"{repo}\")      # add token=... if the repo is private\n"
                 "sys.path.insert(0, path)\nimport clm\n\n"
                 "scorer = clm.Scorer()                       # loads the backbone (needs a CUDA GPU) and the trained heads\n"
                 "print(scorer.rank(\"Question: What is the capital of Japan?\", [\"Tokyo\", \"Osaka\", \"Paris\"]))\n"
                 "# -> {\"pre\": [...], \"mid\": [...], \"post\": [...]}   one probability per option, for each training stage\n```\n")
    lines.append("Or from a terminal inside the downloaded folder: `python 5_demo.py`. A CUDA GPU is required.\n")
    lines.append("**Write situations the way the model was trained:** a passage first when there is one, then "
                 "`Question: ...`. Bare questions with no context are out-of-distribution and score worse.\n")

    lines.append("## Embedding recipe (must match exactly)\n")
    lines.append(f"- Backbone: `{clm.MODEL_NAME}` (frozen)\n"
                 f"- Situation text: `{clm.STATE_MODE}` mode, suffix `{clm.STATE_SUFFIX!r}`\n"
                 f"- Option text prefix: `{clm.ACTION_PREFIX!r}`\n"
                 f"- Pooling: `{clm.POOLING}`; centering: `{clm.CENTER}` (averages saved in `data/centering.npz`)\n"
                 f"- Head: residual MLP (`{clm.HEAD_ARCH}`), width {clm.HEAD_WIDTH}, starts as the identity and learns small corrections\n"
                 f"- Recipe id: `{clm.RECIPE_ID}`\n")

    lines.append("## Training (three stages)\n")
    lines.append("1. **pre** - question/answer pairs with a few hard negatives each.\n"
                 "2. **mid** - the same pairs with up to 10 hard negatives per question (plausible wrong answers).\n"
                 "3. **post** - 60% agent decisions + 40% question/answer replay, temperature frozen.\n")
    lines.append("Loss: bidirectional InfoNCE (see the original CLM work, below).\n")

    lines.append("## Results on held-out pairs (same data distribution as training)\n")
    lines.append("| stage | Q&A (pick 1 of 16) | Tricky Q&A (vs hard negatives) | Agent (pick 1 of 16) |\n|---|---|---|---|")
    for stage, ck in infos.items():
        m = ck.get("metrics", {})
        lines.append(f"| {stage} | {percent(m.get('qa_acc'))} | {percent(m.get('tricky_acc'))} | {percent(m.get('agent_acc'))} |")
    lines.append("\n'Q&A (1 of 16)' uses random wrong answers and is easy. 'Tricky Q&A' uses plausible wrong answers and is the "
                 "number to trust. For public benchmarks, run `python benchmark.py` (RACE, BoolQ, ARC, CommonsenseQA) and paste "
                 "your own numbers here.\n")

    lines.append("## Limitations\n")
    lines.append("- It compares the *meaning* of a situation with the meaning of each option. It does **no step-by-step reasoning**, "
                 "so questions that need arithmetic or noticing that something is *missing* are often wrong, sometimes with high confidence.\n"
                 "- It was trained on a narrow mix (web/Wikipedia-style question answering and agent steps). Other kinds of decisions "
                 "(games, policies, scoring) are out-of-distribution.\n"
                 "- It cannot take several questions at once, and it has no numeric 'score' output: one situation, one question, 2+ options.\n"
                 "- Probabilities are not calibrated. Do not use it for decisions where a wrong answer is costly.\n")

    lines.append("## Data and license notes\n")
    lines.append("The Q&A pairs come from public datasets (by default NVIDIA Nemotron-CC Diverse-QA sample and SQuAD 1.1, which is "
                 "CC BY-SA 4.0), and the agent pairs from `neulab/adp-v2`. **Check each dataset's license before choosing this repo's "
                 "license**, and edit this section to match what you actually used. The backbone is `" + clm.MODEL_NAME + "`; "
                 "check its license too.\n")

    lines.append("## Credit\n")
    lines.append("Inspired by *Contrastive Language Models* (Kwok et al., Stanford / NVIDIA, 2026). This is an independent small-scale "
                 "re-implementation, not the authors' model.\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="the model page to create, e.g. YOUR_USERNAME/mini-clm")
    parser.add_argument("--public", action="store_true", help="make the repo public (default: private)")
    parser.add_argument("--dry-run", action="store_true", help="prepare ./hf_upload but upload nothing")
    parser.add_argument("--license", default=None, help="license id for the model card, e.g. apache-2.0 (left out if not given)")
    parser.add_argument("--token", default=None, help="Hugging Face token (default: your saved login or HF_TOKEN)")
    parser.add_argument("--message", default="Upload mini CLM heads", help="commit message")
    args = parser.parse_args()

    # 1. Which stages exist and are valid?
    infos = {}
    for stage in STAGES:
        if os.path.exists(clm.ckpt_path(stage)):
            infos[stage] = load_checkpoint_info(stage)
    if not infos:
        raise SystemExit("No trained heads found in data/. Run `python 4_train.py --stage all` first.")
    if clm.CENTER and not os.path.exists(clm.CENTER_PATH):
        raise SystemExit(f"{clm.CENTER_PATH} is missing but CENTER is on in clm.py; the model cannot be used without it. Rerun 2_embed.py.")

    # 2. Build the folder to upload (a fixed list of files, so a secret like .env can never go along).
    if os.path.exists(STAGE_DIR):
        shutil.rmtree(STAGE_DIR)
    os.makedirs(os.path.join(STAGE_DIR, "data"))
    for name in CODE_FILES:
        source = os.path.join(HERE, name)
        if os.path.exists(source):
            shutil.copy2(source, os.path.join(STAGE_DIR, name))
        else:
            print(f"(skipping {name}: not found)")
    for stage in infos:
        shutil.copy2(clm.ckpt_path(stage), os.path.join(STAGE_DIR, "data", os.path.basename(clm.ckpt_path(stage))))
    if clm.CENTER:
        shutil.copy2(clm.CENTER_PATH, os.path.join(STAGE_DIR, "data", os.path.basename(clm.CENTER_PATH)))

    config = {"model_name": clm.MODEL_NAME, "recipe": clm.RECIPE, "recipe_id": clm.RECIPE_ID, "head_arch": clm.HEAD_ARCH,
              "head_width": clm.HEAD_WIDTH, "state_mode": clm.STATE_MODE, "state_suffix": clm.STATE_SUFFIX,
              "action_prefix": clm.ACTION_PREFIX, "pooling": clm.POOLING, "center": clm.CENTER, "stages": list(infos)}
    with open(os.path.join(STAGE_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # The repo id is needed for the usage example in the model card; a bare name gets your username added below.
    repo = args.repo
    with open(os.path.join(STAGE_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(model_card(repo, infos, args.license))

    total_mb = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(STAGE_DIR) for f in fs) / 1024 ** 2
    print(f"Prepared {STAGE_DIR} ({total_mb:.1f} MB), stages: {', '.join(infos)}")
    for d, _, fs in os.walk(STAGE_DIR):
        for f in sorted(fs):
            print("  ", os.path.relpath(os.path.join(d, f), STAGE_DIR))
    if args.dry_run:
        print("\nDry run: nothing was uploaded. Open hf_upload/README.md to check the model card, then run again without --dry-run.")
        return

    # 3. Upload.
    try:
        from huggingface_hub import HfApi, get_token
    except ImportError:
        raise SystemExit("Install the Hub library first: pip install huggingface_hub")
    token = args.token or os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit("You are not logged in. Run `hf auth login` (a token with WRITE access), or set the HF_TOKEN variable.")
    api = HfApi()
    try:
        user = api.whoami(token=token)["name"]
    except Exception as error:
        raise SystemExit(f"The token was rejected: {error}")
    if "/" not in repo:
        repo = f"{user}/{repo}"
        with open(os.path.join(STAGE_DIR, "README.md"), "w", encoding="utf-8") as f:     # rewrite with the full repo id
            f.write(model_card(repo, infos, args.license))
    print(f"\nLogged in as {user}. Uploading to {repo} ({'public' if args.public else 'private'}) ...")
    api.create_repo(repo_id=repo, repo_type="model", private=not args.public, exist_ok=True, token=token)
    api.upload_folder(folder_path=STAGE_DIR, repo_id=repo, repo_type="model", commit_message=args.message, token=token)
    print(f"Done: https://huggingface.co/{repo}")
    if not args.public:
        print("The repo is private. Make it public in the repo's Settings on huggingface.co when you are ready, "
              "after checking the license notes in the model card.")


if __name__ == "__main__":
    main()
