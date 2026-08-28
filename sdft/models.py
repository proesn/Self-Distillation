"""Model, tokenizer and LoRA setup for the SDFT entry point."""

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
)

# LoRA on every linear projection of the language model. Vision towers (Qwen3-VL) use
# different module names (`qkv`, `proj`, `linear_fc*`), so they are left untouched.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_base_model(model_name, dtype=torch.bfloat16):
    """Load a checkpoint as a text-generation model.

    Text-only checkpoints (Qwen2.5, Qwen3) resolve through `AutoModelForCausalLM`.
    Vision-language checkpoints (Qwen3-VL) are not in that mapping and resolve through
    `AutoModelForImageTextToText`; the trainer only ever feeds them text.
    """
    model_type = AutoConfig.from_pretrained(model_name).model_type
    if model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        cls = AutoModelForCausalLM
    elif model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES:
        cls = AutoModelForImageTextToText
    else:
        raise ValueError(f"{model_name}: model_type {model_type!r} is neither a causal-LM nor an image-text-to-text architecture")
    print(f"Loading {model_name} ({model_type}) via {cls.__name__}")
    return cls.from_pretrained(model_name, torch_dtype=dtype)


def disable_tokenizer_thinking(tokenizer):
    """Default `enable_thinking=False` in `apply_chat_template` (Qwen3 thinking mode).

    The trainer renders both `prompt` and `teacher_prompt` through this tokenizer, so the
    student is trained and scored in no-think mode. Tokenizers whose template lacks the
    kwarg raise `TypeError`; the fallback calls them unchanged.
    """
    original_apply_chat_template = tokenizer.apply_chat_template

    def apply_chat_template_no_thinking(*args, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        try:
            return original_apply_chat_template(*args, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return original_apply_chat_template(*args, **kwargs)

    tokenizer.apply_chat_template = apply_chat_template_no_thinking
    return tokenizer


def load_tokenizer(model_name, no_thinking=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if no_thinking:
        tokenizer = disable_tokenizer_thinking(tokenizer)
    return tokenizer


def make_lora_config(r=64, lora_alpha=128, lora_dropout=0.05):
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )


def attach_adapters(model, teacher_model, args):
    """Return `(model, teacher_model, peft_config)` ready for `DistilTrainer`.

    Three cases:
      --adapter_path   both copies load the saved adapter (student trainable, teacher frozen);
                       `peft_config` is None so the trainer reuses the PeftModel as-is.
      --use_lora       the teacher is wrapped HERE with a fresh adapter (lora_B = 0, so it
                       equals the base model at step 0); the student is wrapped by the trainer
                       from `peft_config`. Wrapping both gives them identical parameter names,
                       which the EMA sync matches on.
      neither          full fine-tuning; nothing wrapped.
    """
    if args.adapter_path is not None:
        print(f"Loading LoRA adapter from {args.adapter_path} into student (trainable) and teacher (frozen)")
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
        teacher_model = PeftModel.from_pretrained(teacher_model, args.adapter_path, is_trainable=False)
        return model, teacher_model, None
    if args.use_lora:
        peft_config = make_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)
        teacher_model = get_peft_model(teacher_model, peft_config)
        return model, teacher_model, peft_config
    return model, teacher_model, None
