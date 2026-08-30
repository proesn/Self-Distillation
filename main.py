"""SDFT entry point.

    python main.py --dataset_name kkp --model_name Qwen/Qwen3-4B --output_dir <out> \
        --use_lora --max_prompt_length 8192 --max_completion_length 8192 \
        --eval_steps 10 --eval_datasets kkp

Two copies of the checkpoint are loaded: the student (trained) and the teacher (`ref_model`,
scored under the demonstration-augmented `teacher_prompt`, updated as an EMA of the student
every step). Everything not on the CLI is fixed in the `DistilConfig(...)` literal below.
"""

import argparse
import os
import sys

from transformers import set_seed

from distil_config import DistilConfig
from distil_trainer import DistilTrainer
from sdft.data import DATASETS, load_train_dataset, prompt_length_report
from sdft.eval_callback import PeriodicEvalCallback
from sdft.models import attach_adapters, load_base_model, load_tokenizer
from sdft.gpu import wait_gpu_free
from sdft.runlog import CHECKPOINTS_DIR, MetricsCallback, RunRecord, infer_run_dir, instrument_timing, make_run_id


def parse_args():
    parser = argparse.ArgumentParser(description="Distil Trainer")
    # Reference surface
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=float, default=1, help="Number of training epochs")
    parser.add_argument("--num_prompts_per_batch", type=int, default=32, help="Prompts per optimizer step (= gradient accumulation steps at batch size 1)")
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.01, help="EMA rate of the teacher toward the student (0 = frozen teacher)")
    parser.add_argument("--output_dir", type=str, default=None, help="Checkpoint directory (default: checkpoints/<run_id>)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name or path")
    parser.add_argument("--dataset_name", type=str, default="tooluse", choices=DATASETS, help="Training dataset")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    # Loss / lengths / saving
    parser.add_argument("--alpha", type=float, default=0.0, help="Divergence: 0 = forward KL (default), 1 = reverse KL, else generalised JSD")
    parser.add_argument("--max_prompt_length", type=int, default=1024, help="Applies to BOTH prompt views; the teacher prompt includes the demonstration")
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--allow_prompt_truncation", action="store_true", help="Proceed even if some teacher prompts exceed --max_prompt_length (default: abort — truncation is silent and cuts the question, not the demo)")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Recompute activations in backward (fits long sequences on 48 GB cards; ~30% slower)")
    parser.add_argument("--save_lora_adapter_only", action="store_true", help="Checkpoints hold only adapter + tokenizer (no optimizer state; not resumable)")
    # LoRA
    parser.add_argument("--use_lora", action="store_true", help="Train LoRA adapters instead of full fine-tuning")
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--adapter_path", type=str, default=None, help="Continue from a saved adapter (loaded into student and teacher; implies --use_lora)")
    # In-training eval
    parser.add_argument("--eval_steps", type=int, default=0, help="Evaluate every N optimizer steps with the trainer's vLLM (0 disables)")
    parser.add_argument("--eval_datasets", type=str, nargs="+", choices=DATASETS, default=None, help="Eval sets; default: the training dataset")
    parser.add_argument("--eval_max_new_tokens", type=int, default=1024)
    parser.add_argument("--eval_temperature", type=float, default=0.0, help="0 = greedy")
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--eval_num_samples", type=int, default=-1, help="If > 0, evaluate only the first N samples of each eval set")
    # Run record (experiments/runs/<run_id>/) — see sdft/runlog.py
    parser.add_argument("--name", type=str, default=None, help="Short human label; run_id = YYYY-MM-DD_<label> (default: dataset name)")
    parser.add_argument("--group", type=str, default=None, help="Sweep / experiment group the run belongs to")
    parser.add_argument("--tags", type=str, nargs="*", default=[], help="Free tags")
    parser.add_argument("--idea", type=str, nargs="*", default=[], help="Brain idea slugs this run tests")
    parser.add_argument("--hypothesis", type=str, default=None, help="One line written into notes.md")
    parser.add_argument("--no_record", action="store_true", help="Do not write a run record (debug runs)")
    parser.add_argument("--gpu_wait", type=float, default=300, help="Seconds to wait for another process to release the GPU before loading anything (0 = check once)")
    parser.add_argument("--allow_shared_gpu", action="store_true", help="Start even if the GPU is still >10%% occupied after --gpu_wait (default: abort before loading anything)")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3, help="Share of the GPU reserved by the colocated vLLM engine")
    # Tokenizer / logging
    parser.add_argument("--enable_thinking", action="store_true", help="Keep Qwen3 thinking mode on (default: chat template rendered with enable_thinking=False)")
    parser.add_argument("--run_name", type=str, default=None, help="wandb / Trainer run name")
    parser.add_argument("--wandb_project", type=str, default=None, help="Sets WANDB_PROJECT")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    wait_gpu_free(args.allow_shared_gpu, args.gpu_wait)
    run_id = make_run_id(args.name or args.dataset_name)
    args.output_dir = args.output_dir or os.path.join(CHECKPOINTS_DIR, run_id)
    args.run_name = args.run_name or run_id
    if args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    model = load_base_model(args.model_name)
    teacher_model = load_base_model(args.model_name)
    tokenizer = load_tokenizer(args.model_name, no_thinking=not args.enable_thinking)

    dataset = load_train_dataset(args.dataset_name, seed=args.seed)
    if args.dataset_name == "kkp":
        # Puzzle + ~4k-character demonstration + instruction does not fit the reference 1024.
        # Both prompt views are left-truncated at max_prompt_length, so an undersized value
        # silently cuts the system turn and question head out of the teacher prompt.
        for name in ("max_prompt_length", "max_completion_length", "eval_max_new_tokens"):
            if getattr(args, name) == 1024:
                setattr(args, name, 4096)
                print(f"KKP: --{name} left at 1024, using 4096")

    lengths = prompt_length_report(dataset, tokenizer, args.max_prompt_length)
    n_over = lengths["teacher_prompt"]["n_over"]
    if n_over and not args.allow_prompt_truncation:
        raise SystemExit(
            f"{n_over} teacher prompts exceed --max_prompt_length {args.max_prompt_length} "
            f"(max {lengths['teacher_prompt']['max']} tokens) and would be silently left-truncated. "
            "Raise --max_prompt_length or pass --allow_prompt_truncation."
        )

    config = DistilConfig(
        seed=args.seed,
        run_name=args.run_name,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=True,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=True,
        fp16=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.num_prompts_per_batch,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        num_iterations=1,
        num_generations=1,
        save_steps=args.save_steps,
        save_lora_adapter_only=args.save_lora_adapter_only,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        max_grad_norm=1,
        report_to="wandb",
        output_dir=args.output_dir,
        log_completions=False,  # True for debugging
        sync_ref_model=True,
        ref_model_sync_steps=1,
        ref_model_mixup_alpha=args.ref_model_mixup_alpha,
        vllm_importance_sampling_correction=True,
        num_loss_tokens_to_skip=3,
        alpha=args.alpha,
    )

    model, teacher_model, peft_config = attach_adapters(model, teacher_model, args)

    record = None
    callbacks = []
    if not args.no_record:
        parent = infer_run_dir(args.adapter_path)
        record = RunRecord.start(
            run_id, args, config, dataset, args.model_name,
            lora={"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout} if (args.use_lora or args.adapter_path) else None,
            output_dir=args.output_dir, group=args.group, tags=args.tags, idea=args.idea,
            parent_run=os.path.basename(parent) if parent else None,
        )
        record.write_notes_skeleton(args.hypothesis)
        callbacks.append(MetricsCallback(record))

    eval_callback = None
    if args.eval_steps > 0:
        eval_callback = PeriodicEvalCallback(
            eval_dataset_names=args.eval_datasets or [args.dataset_name],
            tokenizer=tokenizer,
            eval_steps=args.eval_steps,
            max_new_tokens=args.eval_max_new_tokens,
            temperature=args.eval_temperature,
            seed=args.eval_seed,
            eval_num_samples=args.eval_num_samples,
        )
        callbacks.append(eval_callback)

    try:
        trainer = DistilTrainer(
            model=model,
            ref_model=teacher_model,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            callbacks=callbacks or None,
        )
        if eval_callback is not None:
            eval_callback.trainer = trainer
        instrument_timing(trainer)
        trainer.train()
    except KeyboardInterrupt:
        if record: record.finalize("killed", error="KeyboardInterrupt")
        raise
    except BaseException as e:  # includes SystemExit from vLLM/torch startup failures
        if record: record.finalize("failed", error=repr(e))
        raise
    if record:
        record.finalize("finished")

    # The colocated vLLM engine + NCCL group tear down badly at interpreter exit (segfault / non-zero
    # status after everything is saved and logged), so the process leaves through os._exit. That skips
    # atexit, where wandb would mark the run finished — do it explicitly.
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
