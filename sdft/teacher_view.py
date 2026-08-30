"""Teacher view: the model under the trainer's teacher prompt vs. under the plain prompt, on the same training items.

SDFT's teacher is the model conditioned on the demonstration. Scoring that view directly shows
what the distillation signal contains: `teacher_accuracy` = model + teacher prompt,
`student_accuracy` = model + plain prompt, both on identical rows, plus the paired counts
(`teacher_only` = rows the teacher view got right and the student view got wrong).
"""

import json
import os

import numpy as np


def run_teacher_view(dataset, rows, generate, score, output_dir, rec=None):
    """rows: [{prompt, teacher_prompt, gold}] from sdft.data.load_teacher_view.

    generate(prompts) -> list[str]        the eval script's generator (chat template applied inside)
    score(responses, golds) -> (scores, extra)   the eval script's scorer; `extra` = per-view metrics such as parse_rate
    Writes teacher_view_results.json / teacher_view_responses.json under output_dir; finishes `rec` if given.
    """
    golds = [r["gold"] for r in rows]
    views = {}
    for view, key in (("teacher", "teacher_prompt"), ("student", "prompt")):
        prompts = [r[key] for r in rows]
        print(f"\n[{view} view] generating {len(prompts)} responses...")
        responses = generate(prompts)
        scores, extra = score(responses, golds)
        views[view] = {"responses": responses, "scores": [int(s) for s in scores], "extra": dict(extra or {})}

    t, s = views["teacher"]["scores"], views["student"]["scores"]
    metrics = {
        "teacher_accuracy": float(np.mean(t)) if t else 0.0,
        "student_accuracy": float(np.mean(s)) if s else 0.0,
        "num_total": len(rows),
        "teacher_only": int(sum(1 for a, b in zip(t, s) if a and not b)),
        "student_only": int(sum(1 for a, b in zip(t, s) if b and not a)),
        "both": int(sum(1 for a, b in zip(t, s) if a and b)),
    }
    metrics["gap"] = metrics["teacher_accuracy"] - metrics["student_accuracy"]
    for view in ("teacher", "student"):
        for k, v in views[view]["extra"].items():
            metrics[f"{view}_{k}"] = v

    print("\n" + "=" * 60)
    print(f"Teacher view ({dataset}, {len(rows)} training items)")
    print(f"  teacher (model + teacher prompt): {metrics['teacher_accuracy']:.4f}")
    print(f"  student (model + plain prompt):   {metrics['student_accuracy']:.4f}")
    print(f"  gap: {metrics['gap']:+.4f}   teacher-only {metrics['teacher_only']} · student-only {metrics['student_only']} · both {metrics['both']}")
    for k, v in metrics.items():
        if k.endswith("_rate"):
            print(f"  {k}: {v:.4f}")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "teacher_view_results.json")
    with open(results_path, "w") as f:
        json.dump({"dataset": dataset, "split": "train", **metrics, "per_sample": {"teacher": t, "student": s}}, f, indent=2)
    responses_path = os.path.join(output_dir, "teacher_view_responses.json")
    with open(responses_path, "w") as f:
        json.dump(
            [
                {
                    "prompt": rows[i]["prompt"],
                    "teacher_prompt": rows[i]["teacher_prompt"],
                    "gold": rows[i]["gold"],
                    "teacher_response": views["teacher"]["responses"][i],
                    "student_response": views["student"]["responses"][i],
                    "teacher_correct": bool(t[i]),
                    "student_correct": bool(s[i]),
                }
                for i in range(len(rows))
            ],
            f,
            indent=2,
        )
    print(f"Saved {results_path} and {responses_path}")
    if rec is not None:
        rec.finish(metrics, per_sample={"teacher": t, "student": s}, responses_path=os.path.abspath(responses_path))
    return metrics
