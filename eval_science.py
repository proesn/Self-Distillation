import argparse
import os
import json
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer

from sdft.gpu import wait_gpu_free

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on science test set")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Optional LoRA adapter to load with vLLM")
    parser.add_argument("--max_lora_rank", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save evaluation results (defaults to model_path)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 for greedy)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for vLLM sampling")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--run_dir", type=str, default=None, help="Training run this eval targets (default: inferred from a checkpoints/<run_id>/ path)")
    parser.add_argument("--name", type=str, default=None, help="Label of the eval record: run id = YYYY-MM-DD_<label> (default: eval-<dataset>-<target|base>)")
    parser.add_argument("--no_record", action="store_true", help="Do not write an eval record")
    parser.add_argument("--num_samples", type=int, default=-1, help="Evaluate only the first N items (eval split) / a seeded N-item subset (teacher view); -1 = all")
    parser.add_argument("--teacher_view", action="store_true", help="Score the model under the trainer's teacher prompt (demonstration in context) and under the plain prompt on the same --num_samples training items")
    parser.add_argument("--gpu_wait", type=float, default=300, help="Seconds to wait for another process to release the GPU before loading anything (0 = check once)")
    parser.add_argument("--allow_shared_gpu", action="store_true", help="Start even if the GPU is still >10%% occupied after --gpu_wait")
    return parser.parse_args()


def load_model_and_tokenizer(
    model_path,
    gpu_memory_utilization=0.8,
    max_model_len=4096,
    adapter_path=None,
    max_lora_rank=128,
):
    """Load model using vLLM and tokenizer from the given path."""
    from vllm import LLM

    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
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


def load_test_data():
    """Load science test dataset."""
    path = os.path.join(SCRIPT_DIR, 'data', 'science_data', 'eval_data')
    print(f"Loading science test dataset from {path}")
    data = Dataset.load_from_disk(path)
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
    max_new_tokens=2048,
    temperature=0.0,
    seed=42,
    adapter_path=None,
):
    """Generate responses from the model using vLLM."""
    from vllm import SamplingParams

    formatted_prompts = []
    for prompt in prompts:
        formatted_prompt = apply_chat_template_no_thinking(tokenizer, prompt)
        formatted_prompts.append(formatted_prompt)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        seed=seed,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None,
    )

    print(f"Generating responses for {len(formatted_prompts)} prompts...")
    lora_request = None
    if adapter_path is not None:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest(
            lora_name="science_eval_lora",
            lora_int_id=1,
            lora_path=adapter_path,
        )
    outputs = llm.generate(
        formatted_prompts,
        sampling_params,
        lora_request=lora_request,
    )
    return [output.outputs[0].text for output in outputs]


def extract_xml_answer(text: str) -> str:
    """Extract answer from XML-formatted text."""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def evaluate_correctness(responses, answers):
    """
    Evaluate if responses match the golden answers (strict <answer>X</answer>).
    Returns list of scores (1 for correct, 0 for incorrect).
    """
    results = []
    for response, answer in zip(responses, answers):
        extracted = extract_xml_answer(response)
        results.append(1 if extracted == answer else 0)
    return results


def main():
    args = parse_args()
    wait_gpu_free(args.allow_shared_gpu, args.gpu_wait)
    rec = None
    if not args.no_record:
        from sdft.runlog import EvalRecord, infer_run_dir

        target = os.path.basename(args.run_dir) if args.run_dir else None
        if target is None:
            d = infer_run_dir(args.adapter_path, args.model_path)
            target = os.path.basename(d) if d else None
        view = "teacher-view-" if args.teacher_view else ""
        rec = EvalRecord.start(
            args.name or f"eval-science-{view}{target or 'base'}", "science", args.model_path, adapter_path=args.adapter_path, target_run=target,
            settings={**{"max_new_tokens": args.max_new_tokens, "temperature": args.temperature, "seed": args.seed, "max_model_len": args.max_model_len},
                      "view": "teacher" if args.teacher_view else "student", "split": "train" if args.teacher_view else "eval"},
            num_samples=args.num_samples if args.num_samples > 0 else None,
        )

    # Load model and data
    llm, tokenizer = load_model_and_tokenizer(
        args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        adapter_path=args.adapter_path,
        max_lora_rank=args.max_lora_rank,
    )
    if args.teacher_view:
        from sdft.data import load_teacher_view
        from sdft.teacher_view import run_teacher_view

        rows = load_teacher_view("science", args.num_samples, args.seed)
        run_teacher_view(
            "science", rows,
            generate=lambda prompts: generate_responses(llm, tokenizer, prompts, args.max_new_tokens, args.temperature, args.seed, args.adapter_path),
            score=lambda responses, golds: (evaluate_correctness(responses, golds), {}),
            output_dir=args.output_dir if args.output_dir else (args.adapter_path or args.model_path),
            rec=rec,
        )
        return
    test_data = load_test_data()
    if args.num_samples > 0:
        test_data = test_data.select(range(min(args.num_samples, len(test_data))))

    prompts = [example['prompt'] for example in test_data]
    answers = [example['answer'] for example in test_data]

    # Generate responses
    responses = generate_responses(
        llm, tokenizer, prompts,
        args.max_new_tokens,
        args.temperature,
        args.seed,
        args.adapter_path,
    )

    # Evaluate correctness
    print("\nEvaluating responses...")
    scores = evaluate_correctness(responses, answers)
    accuracy = np.mean(scores)

    # Print results
    print("\n" + "=" * 60)
    print(f"Evaluation Results:")
    print(f"  Total samples: {len(scores)}")
    print(f"  Correct: {sum(scores)}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("=" * 60)

    # Save results
    output_dir = args.output_dir if args.output_dir else (args.adapter_path or args.model_path)
    os.makedirs(output_dir, exist_ok=True)

    results_to_save = {
        "accuracy": float(accuracy),
        "num_correct": int(sum(scores)),
        "num_total": len(scores),
        "per_sample_scores": scores,
        "config": {
            "model_path": args.model_path,
            "adapter_path": args.adapter_path,
            "max_lora_rank": args.max_lora_rank,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
        }
    }

    output_path = os.path.join(output_dir, "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\nSaved results to {output_path}")

    # Save responses for inspection
    responses_path = os.path.join(output_dir, "eval_responses.json")
    with open(responses_path, "w") as f:
        json.dump([
            {
                "prompt": prompts[i],
                "response": responses[i],
                "answer": answers[i],
                "correct": bool(scores[i])
            }
            for i in range(len(responses))
        ], f, indent=2)
    print(f"Saved responses to {responses_path}")

    if rec is not None:
        rec.finish({"accuracy": float(accuracy), "num_total": len(scores)}, per_sample=[int(s) for s in scores], responses_path=os.path.abspath(responses_path))


if __name__ == "__main__":
    main()
