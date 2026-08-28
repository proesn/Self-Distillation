import argparse
import os
import json
import torch
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer
import re
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on tooluse test set")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--max_new_tokens", type=int, default=1024, 
                        help="Maximum number of tokens to generate")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save evaluation results (defaults to model_path)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 for greedy)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for vLLM sampling")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Optional LoRA adapter to load with vLLM")
    parser.add_argument("--max_lora_rank", type=int, default=128)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_model_len", type=int, default=None)
    return parser.parse_args()


def load_model_and_tokenizer(
    model_path,
    gpu_memory_utilization=0.8,
    max_model_len=None,
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
        llm_kwargs.update({"enable_lora": True, "max_lora_rank": max_lora_rank, "max_loras": 1})
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        **llm_kwargs,
    )
    return llm, tokenizer


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


def load_test_data(tokenizer):
    """Load and prepare tooluse test dataset."""
    data_dir = os.path.join(SCRIPT_DIR, 'data', 'tooluse_data', 'eval_data')
    data = load_from_disk(data_dir).to_list()
    
    # Format prompts
    for example in data:
        example['prompt'] = apply_chat_template_no_thinking(
            tokenizer,
            [{'role': 'user', 'content': example['prompt']}],
        )
    
    return data


def generate_responses(llm, tokenizer, prompts, max_new_tokens=1024, temperature=0.0, seed=42, adapter_path=None):
    """Generate responses from the model using vLLM."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        seed=seed,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None,
    )
    
    print(f"Generating responses for {len(prompts)} prompts...")
    lora_request = None
    if adapter_path is not None:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest(lora_name="tooluse_eval_lora", lora_int_id=1, lora_path=adapter_path)
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    return [output.outputs[0].text for output in outputs]


def extract_actions(text):
    """Extract all actions from model response."""
    return re.findall(r'Action:\s*(\w+)', text)


def extract_action_inputs(text):
    """Extract and merge all action inputs from model response."""
    json_blocks = re.findall(r'Action Input:\s*({.*?})', text, re.DOTALL)
    combined_dict = {}
    for block in json_blocks:
        try:
            parsed = json.loads(block)
            combined_dict.update(parsed)
        except json.JSONDecodeError:
            continue
    return combined_dict


def evaluate_correctness(responses, golden_answers):
    """
    Evaluate if responses match the golden answers.
    Returns list of scores (1 for correct, 0 for incorrect).
    """
    results = []
    
    for response, golden_answer in zip(responses, golden_answers):
        # Extract predicted actions and inputs
        pred_actions = extract_actions(response)
        pred_inputs = extract_action_inputs(response)
        
        # Extract ground truth actions and inputs
        gt_actions = [item['Action'] for item in golden_answer]
        gt_inputs = {}
        for item in golden_answer:
            try:
                gt_inputs.update(json.loads(item['Action_Input']))
            except:
                pass
        
        # Check if both actions and inputs match
        actions_match = Counter(pred_actions) == Counter(gt_actions)
        inputs_match = pred_inputs == gt_inputs
        
        results.append(1 if (actions_match and inputs_match) else 0)
    
    return results


def main():
    args = parse_args()
    
    # Load model and data
    llm, tokenizer = load_model_and_tokenizer(
        args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        adapter_path=args.adapter_path,
        max_lora_rank=args.max_lora_rank,
    )
    test_data = load_test_data(tokenizer)
    
    prompts = [example['prompt'] for example in test_data]
    golden_answers = [example['golden_answer'] for example in test_data]
    
    # Generate responses
    responses = generate_responses(
        llm, tokenizer, prompts,
        args.max_new_tokens,
        args.temperature,
        args.seed,
        adapter_path=args.adapter_path,
    )
    
    # Evaluate correctness
    print("\nEvaluating responses...")
    scores = evaluate_correctness(responses, golden_answers)
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
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
        }
    }
    
    output_path = os.path.join(output_dir, "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\nSaved results to {output_path}")
    
    # Optionally save responses for inspection
    responses_path = os.path.join(output_dir, "eval_responses.json")
    with open(responses_path, "w") as f:
        json.dump([
            {
                "prompt": test_data[i]['prompt'],
                "response": responses[i],
                "golden_answer": golden_answers[i],
                "correct": bool(scores[i])
            }
            for i in range(len(responses))
        ], f, indent=2)
    print(f"Saved responses to {responses_path}")


if __name__ == "__main__":
    main()
