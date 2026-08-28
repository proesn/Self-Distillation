#!/usr/bin/env python3
import argparse
import re
import shutil
from pathlib import Path

from datasets import Dataset, load_dataset


DATASET_NAME = "RedaAlami/knights-knaves-puzzles"
SYSTEM_PROMPT = """You are a careful logic puzzle solver.
Solve the knights-and-knaves puzzle. Knights always tell the truth, and knaves always lie.

Think step by step, by considering whether each person is lying and if that leads to contradiction.
After the reasoning, finish with the final role assignment in this exact style:
**Solution:**
Owen is a knave, Joseph is a knave, and Sofia is a knave.

Replace the names and roles with the correct inhabitants for the given puzzle.
""".strip()

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)
_SOLUTION_MARKER_RE = re.compile(
    r"(?:\*\*)?\s*(?:solution|answer)\s*[:：]\s*(?:\*\*)?",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare train/eval splits for RedaAlami/knights-knaves-puzzles."
    )
    parser.add_argument("--output_dir", type=str, default="data/kkp_data")
    parser.add_argument("--eval_size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def final_solution_text(output_text):
    text = _THINK_BLOCK_RE.sub("", output_text or "")
    text = text.replace("<think>", "").replace("</think>", "").strip()
    marker_matches = list(_SOLUTION_MARKER_RE.finditer(text))
    if marker_matches:
        solution = text[marker_matches[-1].end() :].strip()
    else:
        solution = text.strip()
    return f"**Solution:**\n{solution}"


def sft_output_text(output_text):
    return (
        (output_text or "")
        .replace("<think>", "")
        .replace("</think>", "")
        .strip()
    )


def extract_names(question):
    match = re.search(
        r"You meet\s+\d+\s+inhabitants?:\s*(.*?)\.",
        question,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return []

    names_text = match.group(1).replace("\n", " ")
    names_text = re.sub(r"\s+and\s+", ", ", names_text)
    return [name.strip() for name in names_text.split(",") if name.strip()]


def to_train_row(example):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["input"]},
        ],
        "output_text": sft_output_text(example["output"]),
    }


def to_eval_row(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["input"]},
        ],
        "answer": final_solution_text(example["output"]),
        "answer_text": example["output"],
        "names": extract_names(example["input"]),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    train_path = output_dir / "train_data"
    eval_path = output_dir / "eval_data"

    ds = load_dataset(DATASET_NAME, split="train").shuffle(seed=args.seed)
    eval_size = min(args.eval_size, len(ds))

    eval_raw = ds.select(range(eval_size))
    train_raw = ds.select(range(eval_size, len(ds)))

    train_ds = Dataset.from_list([to_train_row(ex) for ex in train_raw])
    eval_ds = Dataset.from_list([to_eval_row(ex) for ex in eval_raw])

    train_path.parent.mkdir(parents=True, exist_ok=True)
    for path in (train_path, eval_path):
        if path.exists():
            shutil.rmtree(path)
    train_ds.save_to_disk(str(train_path))
    eval_ds.save_to_disk(str(eval_path))

    print(f"Saved train split: {train_path} ({len(train_ds)} rows)")
    print(f"Saved eval split:  {eval_path} ({len(eval_ds)} rows)")


if __name__ == "__main__":
    main()
