"""Dataset loaders for SDFT.

The trainer consumes exactly two columns, both chat-message lists:

    prompt          what the student sees
    teacher_prompt  the same prompt with the demonstration spliced in via a template

The demonstration never becomes a target token sequence; it only conditions the teacher.
Adding a task = one loader returning these two columns, registered in `LOADERS`.
"""

import gzip
import json
import os
from string import Template

from datasets import Dataset, load_from_disk

# Resolved relative to the repo root so the entry point works from any cwd.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("SDFT_DATA_ROOT", os.path.join(_REPO_ROOT, "data"))


def _path(*parts):
    return os.path.join(DATA_ROOT, *parts)


# Reference template (tooluse, science).
TEACHER_TEMPLATE = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

# Knights-and-knaves: same shape, task-specific closing instruction.
KKP_TEACHER_TEMPLATE = Template("""
$orig_content

This is an example for a response to the puzzle:
$output_text

Now answer with a response of your own. Think step by step, then finish with the final role assignment under **Solution:**.
""")


def load_tooluse_dataset(seed=42) -> Dataset:
    """ReAct-style tool calls. Raw columns used: `prompt` (str), `golden_response` (list[str])."""
    train_dir = _path("tooluse_data", "train_data")
    dataset = load_from_disk(train_dir)

    def format_example(example):
        return {
            "prompt": [{"role": "user", "content": example["prompt"]}],
            "teacher_prompt": [
                {
                    "role": "user",
                    "content": TEACHER_TEMPLATE.substitute(
                        orig_content=example["prompt"],
                        output_text="\n".join(example["golden_response"]),
                    ),
                }
            ],
        }

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=seed)
    print(f"Loaded {len(dataset)} tooluse training examples from {train_dir}")
    return dataset


def load_science_dataset(seed=42) -> Dataset:
    """Four-option science MCQ with <reasoning>/<answer> format. Raw columns: `messages` ([system, user]), `output_text`."""
    train_dir = _path("science_data", "train_data")
    dataset = load_from_disk(train_dir)

    def format_example(example):
        return {
            "prompt": example["messages"],
            "teacher_prompt": [
                example["messages"][0],
                {
                    "role": "user",
                    "content": TEACHER_TEMPLATE.substitute(
                        orig_content=example["messages"][1]["content"],
                        output_text=example["output_text"],
                    ),
                },
            ],
        }

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=seed)
    print(f"Loaded {len(dataset)} science training examples from {train_dir}")
    return dataset


# Verified Qwen-generated chains of thought, one per puzzle: rows of {"messages": [system, user, assistant]}.
# The .gz is what the repo tracks; a plain .json next to it is honoured if present.
KKP_TRAIN_CANDIDATES = (
    _path("kkp_data", "train_data", "qwen_train_correct.json.gz"),
    _path("kkp_data", "train_data", "qwen_train_correct.json"),
)


def _read_kkp_rows():
    for path in KKP_TRAIN_CANDIDATES:
        if os.path.exists(path):
            opener = gzip.open if path.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8") as f:
                rows = json.load(f)
            if isinstance(rows, dict):
                rows = list(rows.values())
            print(f"Loaded {len(rows)} KKP rows from {path}")
            return rows
    raise FileNotFoundError(
        "KKP training file not found; expected one of: " + ", ".join(KKP_TRAIN_CANDIDATES)
    )


def load_kkp_dataset(seed=42) -> Dataset:
    """Knights-and-knaves puzzles. The assistant turn of each row is the demonstration."""
    dataset = Dataset.from_list(_read_kkp_rows())

    def format_example(example):
        messages = list(example["messages"])
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        if not assistant_messages:
            raise ValueError("KKP row must contain an assistant message")
        return {
            "prompt": prompt_messages,
            "teacher_prompt": [
                prompt_messages[0],
                {
                    "role": "user",
                    "content": KKP_TEACHER_TEMPLATE.substitute(
                        orig_content=prompt_messages[-1]["content"],
                        output_text=assistant_messages[-1]["content"],
                    ),
                },
            ],
        }

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=seed)
    print(f"Loaded {len(dataset)} KKP training examples")
    return dataset


LOADERS = {
    "tooluse": load_tooluse_dataset,
    "science": load_science_dataset,
    "kkp": load_kkp_dataset,
}
DATASETS = tuple(LOADERS)


def load_train_dataset(name, seed=42) -> Dataset:
    try:
        loader = LOADERS[name]
    except KeyError:
        raise ValueError(f"Unknown dataset {name!r}; choose from {DATASETS}") from None
    return loader(seed=seed)


def prompt_length_report(dataset, tokenizer, max_prompt_length):
    """Token lengths of both prompt views under the chat template; returns a dict of stats.

    The trainer left-truncates both views at `max_prompt_length` without warning, cutting the
    system turn and question head out of an over-long teacher prompt while keeping the
    demonstration. Call this before training and act on `n_over`.
    """
    import numpy as np

    def lengths(column):
        return np.array(
            [len(tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)) for msgs in dataset[column]]
        )

    report = {}
    for column in ("prompt", "teacher_prompt"):
        a = lengths(column)
        report[column] = {
            "mean": float(a.mean()),
            "p99": float(np.percentile(a, 99)),
            "max": int(a.max()),
            "n_over": int((a > max_prompt_length).sum()),
        }
        s = report[column]
        print(
            f"[prompt lengths] {column:14s} mean={s['mean']:.0f} p99={s['p99']:.0f} max={s['max']} "
            f"> {max_prompt_length}: {s['n_over']}/{len(a)} rows"
        )
    return report
