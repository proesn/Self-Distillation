"""In-training evaluation through the trainer's colocated vLLM engine."""

import numpy as np
import torch
from transformers import TrainerCallback


def _kkp_score(responses, answer_items):
    from eval_kkp import evaluate_correctness

    answers = [item["answer"] for item in answer_items]
    names = [item["names"] for item in answer_items]
    scores, _, _ = evaluate_correctness(responses, answers, names)
    return scores


def _build_spec(dataset_name, tokenizer, truncate):
    """Prompts, gold answers and (generator, scorer) for one eval set, from its eval module."""
    if dataset_name == "tooluse":
        import eval_tooluse as m

        data = truncate(m.load_test_data(tokenizer))
        return {
            "prompts": [x["prompt"] for x in data],
            "golden": [x["golden_answer"] for x in data],
            "generator": m.generate_responses,
            "scorer": m.evaluate_correctness,
        }
    if dataset_name == "science":
        import eval_science as m

        data = truncate(m.load_test_data())
        return {
            "prompts": [x["prompt"] for x in data],
            "golden": [x["answer"] for x in data],
            "generator": m.generate_responses,
            "scorer": m.evaluate_correctness,
        }
    if dataset_name == "kkp":
        import eval_kkp as m

        data = truncate(m.load_test_data())
        return {
            "prompts": [x["prompt"] for x in data],
            "golden": [{"answer": x["answer"], "names": x["names"]} for x in data],
            "generator": m.generate_responses,
            "scorer": _kkp_score,
        }
    raise ValueError(f"Unknown dataset for eval: {dataset_name}")


class PeriodicEvalCallback(TrainerCallback):
    """Run one or more eval pipelines every `eval_steps` using the trainer's vLLM.

    Sequence per eval: wake the engine → push the current student weights if they moved
    since the last push → greedy generation on each eval set → score → `trainer.log`
    (`eval/<dataset>_accuracy`) → sleep the engine. A step-0 baseline runs on
    `on_train_begin`. Set `callback.trainer = trainer` after constructing the trainer.
    """

    def __init__(
        self,
        eval_dataset_names,
        tokenizer,
        eval_steps,
        max_new_tokens=1024,
        temperature=0.0,
        seed=42,
        eval_num_samples=-1,
    ):
        self.eval_dataset_names = list(eval_dataset_names)
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.seed = seed
        self.eval_num_samples = eval_num_samples
        self.trainer = None  # set after trainer construction

        def _truncate(data):
            return data if self.eval_num_samples <= 0 else list(data)[: self.eval_num_samples]

        self.eval_specs = {name: _build_spec(name, tokenizer, _truncate) for name in self.eval_dataset_names}

    def on_train_begin(self, args, state, control, **kwargs):
        # Step-0 baseline so the curve starts at the untrained accuracy.
        self._run_eval(state.global_step)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps != 0:
            return
        self._run_eval(state.global_step)

    def _run_eval(self, global_step):
        trainer = self.trainer
        if trainer is None or getattr(trainer, "llm", None) is None:
            return

        was_sleep_mode = trainer.args.vllm_enable_sleep_mode
        if was_sleep_mode:
            torch.cuda.empty_cache()
            trainer.llm.wake_up()

        if global_step != trainer._last_loaded_step:
            trainer._move_model_to_vllm()
            trainer._last_loaded_step = global_step

        metrics = {}
        messages = []
        for dataset_name, spec in self.eval_specs.items():
            responses = spec["generator"](
                trainer.llm,
                self.tokenizer,
                spec["prompts"],
                self.max_new_tokens,
                self.temperature,
                self.seed,
            )
            scores = spec["scorer"](responses, spec["golden"])
            accuracy = float(np.mean(scores))
            metrics[f"eval/{dataset_name}_accuracy"] = accuracy
            messages.append(f"[Eval @ step {global_step}] {dataset_name} accuracy: {accuracy:.4f}")

        if was_sleep_mode:
            trainer.llm.sleep(level=1)

        trainer.log(metrics)
        print("\n" + "\n".join(messages) + "\n")
