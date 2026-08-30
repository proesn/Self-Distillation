"""Smoke test for the run-record system without a GPU: fake a run end to end, then exercise every explog view.

    .venv/bin/python scripts/selftest_explog.py
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import types

TMP = tempfile.mkdtemp(prefix="explog-selftest-")
# pretend we were started by scripts/train.sh, as the real launcher does
os.environ["SDFT_LAUNCH_CMD"] = "scripts/train.sh kkp lr5e-5 --group kkp-lr-sweep"
os.environ["SDFT_LAUNCHER"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "train.sh")
os.environ["PROFILE"] = "a6000"; os.environ["LR"] = "5e-5"; os.environ["WANDB_API_KEY"] = "must-not-be-recorded"
os.environ["SDFT_RUNS_DIR"] = os.path.join(TMP, "runs")
os.environ["SDFT_CHECKPOINTS_DIR"] = os.path.join(TMP, "checkpoints")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset  # noqa: E402

from explog import brain, views  # noqa: E402
from explog.records import load_runs, parse_notes, render_notes  # noqa: E402
from sdft.runlog import EvalRecord, MetricsCallback, RunRecord, infer_run_dir, make_run_id, record_eval  # noqa: E402


def fake_run(label, lr, curve, status="finished", group=None):
    args = argparse.Namespace(dataset_name="kkp", name=label, learning_rate=lr, use_lora=True, lora_r=64, lora_alpha=128,
                              lora_dropout=0.05, adapter_path=None, no_record=False, seed=42)
    config = types.SimpleNamespace(learning_rate=lr, num_train_epochs=1.0, max_prompt_length=8192, max_completion_length=8192,
                                   alpha=0.0, sync_ref_model=True, ref_model_mixup_alpha=0.01, seed=42, gradient_accumulation_steps=32)
    ds = Dataset.from_list([{"prompt": [{"role": "user", "content": "q"}], "teacher_prompt": [{"role": "user", "content": "q+demo"}]}] * 3)
    run_id = make_run_id(label)
    rec = RunRecord.start(run_id, args, config, ds, "Qwen/Qwen3-4B", lora={"r": 64, "alpha": 128, "dropout": 0.05},
                          output_dir=os.path.join(os.environ["SDFT_CHECKPOINTS_DIR"], run_id), group=group, tags=["selftest"], idea=["opsd-kkp"])
    rec.write_notes_skeleton("does lr matter")
    cb = MetricsCallback(rec)
    state = types.SimpleNamespace(global_step=0, epoch=0.0)
    for step, acc in curve:
        state.global_step, state.epoch = step, step / 310
        cb.on_log(None, state, None, logs={"loss": 1.0 / (step + 1), "kl_approx": 0.1, "entropy": 0.5})
        cb.on_log(None, state, None, logs={"eval/kkp_accuracy": acc})
    if status == "finished":
        cb.on_train_end(None, state, None)
    else:
        rec.finalize(status, error="boom")
    return rec


def main():
    a = fake_run("lr5e-5", 5e-5, [(0, 0.43), (10, 0.50), (20, 0.55), (30, 0.53)], group="kkp-lr-sweep")
    b = fake_run("lr1e-4", 1e-4, [(0, 0.43), (10, 0.48), (20, 0.47)], group="kkp-lr-sweep")
    c = fake_run("crash", 5e-5, [(0, 0.43)], status="failed")

    # standalone eval = its own record, linked to the training run via the checkpoint path
    ckpt = os.path.join(a.data["output_dir"], "checkpoint-30")
    os.makedirs(ckpt, exist_ok=True)
    assert infer_run_dir(ckpt) == a.dir, (infer_run_dir(ckpt), a.dir)
    ev = EvalRecord.start("eval-science-lr5e-5-s30", "science", "Qwen/Qwen3-4B", adapter_path=ckpt, settings={"temperature": 0.0}, num_samples=507)
    assert ev.data["target"]["run"] == a.run_id and ev.data["target"]["checkpoint_step"] == 30, ev.data["target"]
    ev.finish({"accuracy": 0.61, "num_total": 507}, per_sample=[1, 0, 1], responses_path="/tmp/x.json")
    base = EvalRecord.start("eval-kkp-base", "kkp", "Qwen/Qwen3-4B", settings={"temperature": 0.0}, num_samples=300)
    assert base.data["target"]["run"] is None
    base.finish({"accuracy": 0.57, "parse_rate": 0.9, "num_total": 300})
    record_eval(a.dir, "tooluse", {"accuracy": 0.3, "num_total": 97}, checkpoint=ckpt)  # legacy pointer without an eval record still works

    # judgement
    notes_path = os.path.join(b.dir, "notes.md")
    f = parse_notes(open(notes_path).read())
    f.update(validity="invalid", reason="eval ran on 100 samples", verdict="discard")
    open(notes_path, "w").write(render_notes(f))
    f = parse_notes(open(os.path.join(a.dir, "notes.md")).read())
    f.update(validity="valid", verdict="lr 5e-5 best of sweep", body="Looks clean.")
    open(os.path.join(a.dir, "notes.md"), "w").write(render_notes(f))

    runs = load_runs()
    assert len(runs) == 5, [r.id for r in runs]
    evals = [r for r in runs if r.kind == "eval"]
    assert len(evals) == 2 and {e.summary()["final"] for e in evals} == {0.61, 0.57}, [(e.id, e.summary()) for e in evals]
    ra = next(r for r in runs if r.id == a.run_id)
    s = ra.summary()
    assert s["acc0"] == 0.43 and s["best"] == 0.55 and s["best_step"] == 20 and s["final"] == 0.53 and abs(s["delta"] - 0.10) < 1e-9, s
    assert ra.standalone()["science"]["accuracy"] == 0.61 and ra.results[0]["eval_run"] == ev.run_id, ra.results
    assert next(r for r in runs if r.id == c.run_id).status == "failed"
    assert os.path.exists(os.path.join(a.dir, "code.patch"))
    launch = open(os.path.join(a.dir, "launch.sh")).read()
    assert "git worktree add" in launch and "--name" in launch and "-rerun" in launch and launch.startswith("#!/usr/bin/env bash"), launch
    assert os.path.exists(os.path.join(a.dir, "launcher.sh")), "launcher copy missing"
    assert json.load(open(a.path))["launch"]["invocation"].startswith("scripts/train.sh"), json.load(open(a.path))["launch"]
    print(launch)
    assert json.load(open(a.path))["git"]["sha"]

    index_path, n = views.write_index(path=os.path.join(TMP, "INDEX.md"), runs=runs)
    index = open(index_path).read()
    assert "~~`" in index and "kkp-lr-sweep" in index and "Evaluations (standalone)" in index and "(base)" in index and "55.0 (20)" in index, index
    print(index)
    print(views.show(ra))
    print(views.compare([ra, next(r for r in runs if r.id == b.run_id)]))
    problems = views.check(runs)
    assert all("pending" in msg and rid in (ev.run_id, base.run_id) for rid, msg in problems), problems  # only the unjudged evals are flagged
    print("check: ok")

    brain_dir = os.path.join(TMP, "brain-experiments")
    os.makedirs(os.path.join(brain_dir, "runs"))
    written, idx = brain.sync(runs, brain_experiments_dir=brain_dir)
    ledger = open(written[0]).read()
    assert ledger.startswith("---\nid: ") and "validity:" in ledger and "metrics: {" in ledger, ledger
    print(ledger)
    print(open(idx).read())
    print("SELFTEST OK")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
