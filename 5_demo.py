"""Step 5 - use the trained model (inference).

A System One model does not write text. You give it a situation and a list of options, and it
returns a probability for every option in one pass.

By default it shows the answer of ALL THREE training stages side by side, so you can see what each
stage learned. A "*" marks the option a stage picks.

Run the built-in examples (Q&A and agent decisions):
    python 5_demo.py

Your own situation and options (repeat --option for each choice):
    python 5_demo.py --state "Question: What is the capital of Japan?" --option "Tokyo" --option "Osaka" --option "Paris"

Type situations and options yourself, one after another (the model stays loaded, so it is fast):
    python 5_demo.py --interactive

Many cases from a JSON file: a list of {"state": "...", "options": ["...", "..."]}
    python 5_demo.py --file my_cases.json

Only one stage:
    python 5_demo.py --stage post        # pre, mid or post

From your own Python code:
    import clm
    scorer = clm.Scorer()
    scorer.rank("Question: ...", ["answer A", "answer B"])

Tip: stages 1 and 2 only saw Q&A written like  "Question: ..."  with plain-text answers. Stage 3 also saw
agent decisions written like  tool_name({json arguments}),  so agent options in that style score best.
"""
import argparse
import json

import clm

EXAMPLES = [
    {   # Q&A
        "state": "Question: Who wrote the play Romeo and Juliet?",
        "options": ["William Shakespeare", "Charles Dickens", "Mark Twain", "Jane Austen"],
    },
    {   # Q&A
        "state": "Question: What gas do plants take in from the air for photosynthesis?",
        "options": ["Carbon dioxide", "Oxygen", "Helium", "Methane"],
    },
    {   # agent
        "state": "You are a coding agent working in a Python repo. The test test_parse_date fails with "
                 "ValueError: time data '2024/01/05' does not match format '%Y-%m-%d'. You have not looked at any code yet.",
        "options": [
            'bash({"command": "grep -rn strptime src/"})',
            'bash({"command": "rm -rf tests/"})',
            "I think the answer is 42.",
            'bash({"command": "git push --force origin main"})',
        ],
    },
    {   # agent
        "state": "Task: find out how many open issues the repo has. You are on the repository's GitHub issues page, "
                 "which shows a tab labelled 'Open 37' and a tab labelled 'Closed 412'.",
        "options": [
            'finish({"answer": "37"})',
            'click({"element": "Closed 412"})',
            'type({"text": "hello world"})',
        ],
    },
]


def safe(text, n):
    """One short line that can always be printed, even on a Windows console."""
    return " ".join(str(text).split())[:n].encode("ascii", "replace").decode()


def show(state, options, result):
    """Print one table: a row per option, a column per stage."""
    print("\nSituation:", safe(state, 200))
    stages = list(result)
    print(f"  {'option':<62}" + "".join(f"{s:>9}" for s in stages))
    top = {s: max(range(len(options)), key=result[s].__getitem__) for s in stages}    # which option each stage picks
    for i, option in enumerate(options):
        cells = "".join(f"{result[s][i]:8.1%}{'*' if top[s] == i else ' '}" for s in stages)
        print(f"  {safe(option, 60):<62}{cells}")


def read_interactive():
    """Ask for situations and options on the keyboard. Yields (state, options)."""
    print("Type a situation and press Enter. Press Enter on an empty line to quit.")
    while True:
        state = input("\nSituation> ").strip()
        if not state:
            return
        options = []
        print("Now the options, one per line. Press Enter on an empty line when done.")
        while True:
            option = input(f"  option {len(options) + 1}> ").strip()
            if not option:
                break
            options.append(option)
        if len(options) < 2:
            print("Give at least 2 options.")
            continue
        yield state, options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pre", "mid", "post", "all"], default="all",
                        help="which trained heads to use (default: all three, side by side)")
    parser.add_argument("--state", help="the situation, as plain text")
    parser.add_argument("--option", action="append", help="one possible action; repeat for several")
    parser.add_argument("--interactive", action="store_true", help="type situations and options one after another")
    parser.add_argument("--file", help="JSON file: a list of {\"state\": ..., \"options\": [...]}")
    args = parser.parse_args()

    scorer = clm.Scorer(("pre", "mid", "post") if args.stage == "all" else (args.stage,))
    print(f"Stages loaded: {', '.join(scorer.heads)}.  (* = the option that stage picks)")

    if args.interactive:
        cases = read_interactive()
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            cases = [(c["state"], c["options"]) for c in json.load(f)]
    elif args.state and args.option:
        cases = [(args.state, args.option)]
    else:
        cases = [(c["state"], c["options"]) for c in EXAMPLES]

    for state, options in cases:
        show(state, options, scorer.rank(state, options))


if __name__ == "__main__":
    main()
