"""Support package for the SDFT entry point.

Modules:
    data          — dataset loaders; every loader returns a `datasets.Dataset` with exactly
                    two chat-message-list columns, `prompt` and `teacher_prompt`.
    models        — base-model loading (text or vision-language checkpoints), tokenizer
                    setup, LoRA attachment for student and teacher.
    eval_callback — `PeriodicEvalCallback`: in-training accuracy on the eval sets, run
                    through the trainer's own colocated vLLM engine.
"""
