import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer


DATASET_NAME = "RedaAlami/knights-knaves-puzzles"
SPLIT = "eval_data"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EVAL_PATH = str(SCRIPT_DIR / "data/kkp_data/eval_data")

SYSTEM_PROMPT = """You are a careful logic puzzle solver.
Solve the knights-and-knaves puzzle. Knights always tell the truth, and knaves always lie.

Think step by step, by considering whether each person is lying and if that leads to contradiction.
After the reasoning, finish with the final role assignment in this exact style:
**Solution:**
Owen is a knave, Joseph is a knave, and Sofia is a knave.

Replace the names and roles with the correct inhabitants for the given puzzle.
""".strip()

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)
_ANSWER_BLOCK_RE = re.compile(
    r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL
)
_SOLUTION_MARKER_RE = re.compile(
    r"(?:\*\*)?\s*(?:solution|answer)\s*[:：]\s*(?:\*\*)?",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Evaluate a model on {DATASET_NAME} ({SPLIT})"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--max_lora_rank", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Use first N samples from the dataset split (-1 = full split)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_EVAL_PATH,
        help="Local KKP eval dataset saved with Dataset.save_to_disk().",
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument(
        "--print_samples",
        type=int,
        default=5,
        help="Print the first N prompts/responses for inspection",
    )
    return parser.parse_args()


def load_model_and_tokenizer(
    model_path,
    gpu_memory_utilization=0.8,
    max_model_len=4096,
    adapter_path=None,
    max_lora_rank=128,
):
    from vllm import LLM

    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    llm_kwargs = {}
    if adapter_path is not None:
        print(f"Loading LoRA adapter from {adapter_path}")
        llm_kwargs.update(
            {
                "enable_lora": True,
                "max_lora_rank": max_lora_rank,
                "max_loras": 1,
            }
        )
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=torch.bfloat16,
        max_model_len=max_model_len,
        trust_remote_code=True,
        **llm_kwargs,
    )
    return llm, tokenizer


def strip_think_blocks(text):
    if text is None:
        return ""
    text = _THINK_BLOCK_RE.sub("", text)
    return (
        text.replace("<think>", "")
        .replace("</think>", "")
        .replace("<end_of_turn>", "")
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
    return [
        name.strip()
        for name in names_text.split(",")
        if name.strip()
    ]


def answer_region(text):
    text = strip_think_blocks(text)
    answer_match = _ANSWER_BLOCK_RE.search(text)
    if answer_match:
        return answer_match.group(1).strip()

    marker_matches = list(_SOLUTION_MARKER_RE.finditer(text))
    if marker_matches:
        return text[marker_matches[-1].end() :].strip()

    return text[-1500:].strip()


def extract_assignment(text, names):
    region = answer_region(text)
    assignments = {}

    for name in names:
        escaped = re.escape(name)
        patterns = [
            rf"\b{escaped}\b\s*(?::|-|=)\s*(?:a\s+)?(knight|knave)\b",
            rf"\b{escaped}\b\s+is\s+(?:a\s+)?(knight|knave)\b",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, region, flags=re.IGNORECASE):
                assignments[name] = match.group(1).lower()

    if len(assignments) == len(names):
        return assignments
    return assignments


def format_assignment(assignment, names):
    if not assignment:
        return ""
    return ", ".join(f"{name}: {assignment.get(name, '?')}" for name in names)


def load_test_data(num_samples=-1, data_path=DEFAULT_EVAL_PATH):
    if not data_path or not os.path.exists(data_path):
        raise FileNotFoundError(
            f"KKP eval dataset not found at {data_path}. "
            "Run scripts/prepare_kkp_data.py first."
        )

    print(f"Loading KKP eval data from {data_path}")
    ds = load_from_disk(data_path)

    if num_samples is not None and num_samples > 0:
        ds = ds.select(range(min(num_samples, len(ds))))

    data = []
    for ex in ds:
        prompt = ex["prompt"]
        question = prompt[-1]["content"]
        names = list(ex["names"]) if "names" in ex and ex["names"] else extract_names(question)
        answer_text = ex["answer"]

        answer = extract_assignment(answer_text, names)
        data.append(
            {
                "prompt": prompt,
                "question": question,
                "answer": answer,
                "answer_text": answer_text,
                "names": names,
            }
        )
    return data


def apply_chat_template_no_thinking(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def generate_responses(
    llm,
    tokenizer,
    prompts,
    max_new_tokens=1024,
    temperature=0.0,
    seed=42,
    top_p=0.95,
    top_k=64,
    adapter_path=None,
):
    from vllm import SamplingParams

    formatted_prompts = [
        apply_chat_template_no_thinking(tokenizer, prompt) for prompt in prompts
    ]
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
        seed=seed,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None,
    )
    print(f"Generating responses for {len(formatted_prompts)} prompts...")
    lora_request = None
    if adapter_path is not None:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest(
            lora_name="kkp_eval_lora",
            lora_int_id=1,
            lora_path=adapter_path,
        )
    outputs = llm.generate(
        formatted_prompts,
        sampling_params,
        lora_request=lora_request,
    )
    return [output.outputs[0].text for output in outputs]


def evaluate_correctness(responses, answers, names_list):
    scores = []
    predictions = []
    parse_success = []

    for response, gold, names in zip(responses, answers, names_list):
        pred = extract_assignment(response, names)
        predictions.append(pred)
        parse_success.append(1 if len(pred) == len(names) else 0)
        scores.append(1 if pred == gold and len(pred) == len(names) else 0)

    return scores, predictions, parse_success


def print_sample_outputs(eval_data, responses, predictions, scores, max_samples=5):
    n = min(max_samples, len(responses))
    if n <= 0:
        return

    print("\n" + "=" * 60)
    print(f"Sample model outputs ({n})")
    print("=" * 60)
    for i in range(n):
        names = eval_data[i]["names"]
        print(f"\n--- sample {i} ---")
        print("[PUZZLE]")
        print(eval_data[i]["question"])
        print("\n[GOLD]")
        print(format_assignment(eval_data[i]["answer"], names))
        print("\n[MODEL RAW RESPONSE]")
        print(responses[i])
        print("\n[PARSED / SCORE]")
        print(f"prediction={format_assignment(predictions[i], names)!r}")
        print(f"correct={bool(scores[i])}")


def main():
    args = parse_args()

    llm, tokenizer = load_model_and_tokenizer(
        args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        adapter_path=args.adapter_path,
        max_lora_rank=args.max_lora_rank,
    )
    eval_data = load_test_data(args.num_samples, data_path=args.data_path)

    prompts = [ex["prompt"] for ex in eval_data]
    answers = [ex["answer"] for ex in eval_data]
    names_list = [ex["names"] for ex in eval_data]

    responses = generate_responses(
        llm,
        tokenizer,
        prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
        top_p=args.top_p,
        top_k=args.top_k,
        adapter_path=args.adapter_path,
    )

    print("\nEvaluating responses...")
    scores, predictions, parse_success = evaluate_correctness(
        responses, answers, names_list
    )
    accuracy = float(np.mean(scores)) if scores else 0.0
    parse_rate = float(np.mean(parse_success)) if parse_success else 0.0

    print_sample_outputs(
        eval_data,
        responses,
        predictions,
        scores,
        max_samples=args.print_samples,
    )

    print("\n" + "=" * 60)
    print("Knights-and-Knaves Results:")
    print(f"  Dataset: {DATASET_NAME} ({SPLIT})")
    print(f"  Total samples: {len(scores)}")
    print(f"  Correct: {sum(scores)}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"  Parsed complete assignments: {sum(parse_success)}")
    print(f"  Parse rate: {parse_rate:.4f} ({parse_rate * 100:.2f}%)")
    print("=" * 60)

    output_dir = args.output_dir if args.output_dir else (args.adapter_path or args.model_path)
    os.makedirs(output_dir, exist_ok=True)

    results_to_save = {
        "dataset": DATASET_NAME,
        "split": SPLIT,
        "accuracy": accuracy,
        "parse_rate": parse_rate,
        "num_correct": int(sum(scores)),
        "num_total": len(scores),
        "per_sample_scores": scores,
        "parse_success": parse_success,
        "config": {
            "model_path": args.model_path,
            "adapter_path": args.adapter_path,
            "max_lora_rank": args.max_lora_rank,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
            "num_samples": args.num_samples,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
        },
    }

    results_path = os.path.join(output_dir, "eval_kkp_results.json")
    with open(results_path, "w") as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\nSaved results to {results_path}")

    responses_path = os.path.join(output_dir, "eval_kkp_responses.json")
    with open(responses_path, "w") as f:
        json.dump(
            [
                {
                    "prompt": prompts[i],
                    "question": eval_data[i]["question"],
                    "names": names_list[i],
                    "response": responses[i],
                    "prediction": predictions[i],
                    "answer": answers[i],
                    "answer_text": eval_data[i]["answer_text"],
                    "correct": bool(scores[i]),
                    "parse_success": bool(parse_success[i]),
                }
                for i in range(len(responses))
            ],
            f,
            indent=2,
        )
    print(f"Saved responses to {responses_path}")


if __name__ == "__main__":
    main()
