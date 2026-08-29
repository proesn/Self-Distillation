"""Run records — one immutable folder per experiment run.

    experiments/runs/<run_id>/
        run.json       provenance + resolved config (written at launch; status/end patched at exit)
        events.jsonl   metric stream from the trainer, one JSON object per log step
        results.json   standalone evaluation results, appended by the eval scripts
        code.patch     `git diff HEAD` at launch (empty when the tree was clean)
        notes.md       human judgement: validity / reason / verdict / idea + free text

Everything a human would forget is written here by the code; humans write only notes.md.
Views (tables, comparisons, the brain ledger) are generated from these files by `explog`.
"""

import atexit
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata

from transformers import TrainerCallback

SCHEMA_VERSION = 1
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.environ.get("SDFT_RUNS_DIR", os.path.join(REPO_ROOT, "experiments", "runs"))
CHECKPOINTS_DIR = os.environ.get("SDFT_CHECKPOINTS_DIR", os.path.join(REPO_ROOT, "checkpoints"))

# DistilConfig fields worth recording (the CLI args are recorded in full separately).
CONFIG_FIELDS = (
    "learning_rate", "num_train_epochs", "per_device_train_batch_size", "gradient_accumulation_steps",
    "warmup_ratio", "lr_scheduler_type", "max_grad_norm", "seed",
    "max_prompt_length", "max_completion_length", "temperature", "num_generations",
    "alpha", "beta", "sync_ref_model", "ref_model_sync_steps", "ref_model_mixup_alpha",
    "generate_from_teacher", "num_loss_tokens_to_skip", "vllm_importance_sampling_correction",
    "vllm_importance_sampling_cap", "vllm_gpu_memory_utilization", "save_steps", "save_lora_adapter_only",
    "gradient_checkpointing",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slugify(label):
    s = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return s or "run"


def make_run_id(label, date=None):
    """`YYYY-MM-DD_<label>`; a numeric suffix is added only if the folder already exists."""
    date = date or datetime.now().strftime("%Y-%m-%d")
    base = f"{date}_{slugify(label)}"
    run_id, n = base, 2
    while os.path.exists(os.path.join(RUNS_DIR, run_id)):
        run_id = f"{base}-{n}"
        n += 1
    return run_id


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    except Exception:
        return ""


def git_state():
    porcelain = _git("status", "--porcelain")
    untracked = [line[3:] for line in porcelain.splitlines() if line.startswith("??")]
    return {
        "sha": _git("rev-parse", "HEAD").strip() or None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD").strip() or None,
        "dirty": bool(porcelain.strip()),
        "untracked": untracked,
    }


def package_versions():
    out = {"python": platform.python_version()}
    for name in ("torch", "transformers", "trl", "peft", "vllm", "datasets", "accelerate"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def gpu_name():
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def gpu_memory():
    """Peak allocated / currently reserved GPU memory of this process (GB); vLLM colocated in-process is included."""
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "gpu_mem_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "gpu_mem_reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 2),
            }
    except Exception:
        pass
    return {}


def model_revision(model_name):
    """Commit hash of the HF model repo, from the hub if reachable, else from the local cache."""
    try:
        from huggingface_hub import model_info

        return model_info(model_name, timeout=5).sha
    except Exception:
        pass
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        ref = os.path.join(HF_HUB_CACHE, "models--" + model_name.replace("/", "--"), "refs", "main")
        with open(ref) as f:
            return f.read().strip()
    except Exception:
        return None


def dataset_fingerprint(dataset):
    return {"rows": len(dataset), "fingerprint": getattr(dataset, "_fingerprint", None)}


# Environment that shapes a run and is safe to record (never tokens/keys).
ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES", "PROFILE", "LR", "EPOCHS", "SAVE_STEPS", "EVAL_STEPS", "MODEL",
    "WANDB_PROJECT", "WANDB_MODE", "WANDB_RUN_GROUP", "HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
    "PYTHONPATH", "CONDA_DEFAULT_ENV", "VIRTUAL_ENV", "OMP_NUM_THREADS",
)
ENV_PREFIXES = ("VLLM_", "SDFT_", "TORCH_", "NCCL_")
ENV_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def launch_env():
    out = {}
    for k, v in os.environ.items():
        if any(m in k.upper() for m in ENV_SECRET_MARKERS):
            continue
        if k in ENV_KEYS or k.startswith(ENV_PREFIXES):
            out[k] = v
    return dict(sorted(out.items()))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def launch_info():
    """How this process was started: the launcher script (if it exported SDFT_LAUNCH_CMD), env, python, argv."""
    launcher = os.environ.get("SDFT_LAUNCHER")
    return {
        "invocation": os.environ.get("SDFT_LAUNCH_CMD"),        # what was typed, e.g. `scripts/train.sh kkp lr5e-5 --group x`
        "launcher": launcher,                                     # path of that script
        "launcher_sha256": _sha256(launcher) if launcher and os.path.exists(launcher) else None,
        "python": sys.executable,
        "cwd": os.getcwd(),
        "argv": list(sys.argv),
        "env": launch_env(),
    }


def write_launch_script(run_dir, info, git):
    """launch.sh: re-run this exact run — same code state, env and command — under a new name."""
    argv = list(info["argv"])
    if "--name" in argv:
        i = argv.index("--name")
        argv[i + 1] = argv[i + 1] + "-rerun"
    else:
        argv += ["--name", os.path.basename(run_dir) + "-rerun"]
    env_lines = "\n".join(f"export {k}={shlex.quote(v)}" for k, v in info["env"].items())
    script = f"""#!/usr/bin/env bash
# Re-run of {os.path.basename(run_dir)} — generated at launch by sdft/runlog.py.
#   typed:   {info.get('invocation') or '(python invoked directly)'}
#   host:    {socket.gethostname()}    cwd: {info['cwd']}
#   python:  {info['python']}
# Usage: bash experiments/runs/{os.path.basename(run_dir)}/launch.sh [--name <label>] [extra main.py args]
#   Runs at the recorded code state. If your checkout differs (other commit, dirty tree, or the run
#   had a code.patch), it runs inside a throwaway worktree under .rerun-worktrees/ — your checkout
#   is never touched. SDFT_RERUN_SAME_CODE=0 runs your current code instead.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"; cd "$ROOT"
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
SHA={git.get('sha') or 'HEAD'}
if [ "${{SDFT_RERUN_SAME_CODE:-1}}" = 1 ] && {{ [ "$(git rev-parse HEAD)" != "$SHA" ] || [ -n "$(git status --porcelain)" ] || [ -s "$HERE/code.patch" ]; }}; then
  WT="$ROOT/.rerun-worktrees/${{SHA:0:10}}"
  [ -d "$WT" ] || git worktree add --detach --quiet "$WT" "$SHA"
  git -C "$WT" checkout --quiet -- .
  [ -s "$HERE/code.patch" ] && git -C "$WT" apply "$HERE/code.patch"
  echo "[launch.sh] running in worktree $WT at $SHA$([ -s "$HERE/code.patch" ] && echo ' + code.patch')"
  cd "$WT"
fi
{env_lines}
export SDFT_RUNS_DIR="${{SDFT_RUNS_DIR:-$ROOT/experiments/runs}}"
export SDFT_CHECKPOINTS_DIR="${{SDFT_CHECKPOINTS_DIR:-$ROOT/checkpoints}}"
exec "${{PYTHON:-{info['python']}}}" {shlex.join(argv[0:1])} {shlex.join(argv[1:])} "$@"
"""
    path = os.path.join(run_dir, "launch.sh")
    with open(path, "w") as f:
        f.write(script)
    os.chmod(path, 0o755)
    return path


def _dump(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
    os.replace(tmp, path)


class RunRecord:
    """Writes run.json / code.patch at launch and patches status at exit."""

    def __init__(self, run_id):
        self.run_id = run_id
        self.dir = os.path.join(RUNS_DIR, run_id)
        self.path = os.path.join(self.dir, "run.json")
        self.data = None

    @classmethod
    def start(cls, run_id, args, config, dataset, model_name, lora=None, output_dir=None, group=None, tags=(), idea=(), parent_run=None):
        rec = cls(run_id)
        os.makedirs(rec.dir, exist_ok=True)
        git = git_state()
        with open(os.path.join(rec.dir, "code.patch"), "w") as f:
            f.write(_git("diff", "HEAD"))
        launch = launch_info()
        write_launch_script(rec.dir, launch, git)
        if launch["launcher"] and os.path.exists(launch["launcher"]):
            shutil.copyfile(launch["launcher"], os.path.join(rec.dir, "launcher.sh"))  # verbatim copy of the script that started this run
        rec.data = {
            "schema_version": SCHEMA_VERSION,
            "id": run_id,
            "name": getattr(args, "name", None) or run_id,
            "group": group,
            "tags": list(tags),
            "idea": list(idea),
            "parent_run": parent_run,
            "status": "running",
            "started": utc_now(),
            "ended": None,
            "host": socket.gethostname(),
            "gpu": gpu_name(),
            "git": git,
            "cmd": shlex.join(sys.argv),
            "launch": launch,
            "args": {k: v for k, v in vars(args).items()},
            "config": {k: getattr(config, k, None) for k in CONFIG_FIELDS},
            "model": {"name": model_name, "revision": model_revision(model_name)},
            "data": {"name": getattr(args, "dataset_name", None), **dataset_fingerprint(dataset)},
            "lora": lora,
            "env": package_versions(),
            "output_dir": output_dir,
            "wandb_url": None,
            "error": None,
        }
        rec.save()
        # Fallback if the process dies without reaching finalize(): mark it, never leave "running".
        atexit.register(rec._atexit)
        print(f"[runlog] recording to {rec.dir}")
        return rec

    def save(self):
        _dump(self.path, self.data)

    def set(self, **fields):
        self.data.update(fields)
        self.save()

    def finalize(self, status, error=None):
        if self.data.get("status") != "running":
            return
        self.data["status"] = status
        self.data["ended"] = utc_now()
        if error:
            self.data["error"] = str(error)[:2000]
        self.save()
        print(f"[runlog] {self.run_id}: {status}")

    def _atexit(self):
        if self.data and self.data.get("status") == "running":
            self.finalize("killed", error="process exited without finalize()")

    def write_notes_skeleton(self, hypothesis=""):
        path = os.path.join(self.dir, "notes.md")
        if os.path.exists(path):
            return
        with open(path, "w") as f:
            f.write("validity: pending\nreason:\nverdict:\nidea:\n---\n")
            if hypothesis:
                f.write(f"**Hypothesis:** {hypothesis}\n")


class MetricsCallback(TrainerCallback):
    """Append every Trainer log line to events.jsonl (loss, kl_approx, entropy, eval/*_accuracy, lr, …)."""

    def __init__(self, record):
        self.record = record
        self.path = os.path.join(record.dir, "events.jsonl")

    def on_train_begin(self, args, state, control, **kwargs):
        # WandbCallback (a default callback, so it runs before this one) has initialised the run by now.
        try:
            import wandb

            if wandb.run is not None:
                self.record.set(wandb_url=wandb.run.url)
        except Exception:
            pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        row = {"step": state.global_step, "epoch": state.epoch, "time": utc_now()}
        row.update({k: v for k, v in logs.items() if isinstance(v, (int, float, str)) or v is None})
        row.update(gpu_memory())
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def on_train_end(self, args, state, control, **kwargs):
        peak = gpu_memory().get("gpu_mem_peak_gb")
        if peak is not None:
            self.record.set(peak_gpu_mem_gb=peak)
        self.record.finalize("finished")


# ---------- results.json (standalone evaluations) ----------

def infer_run_dir(*paths):
    """Find the run a checkpoint belongs to: the segment after `checkpoints/` is the run id."""
    for p in paths:
        if not p:
            continue
        parts = os.path.normpath(os.path.abspath(p)).split(os.sep)
        if "checkpoints" in parts:
            i = parts.index("checkpoints")
            if i + 1 < len(parts):
                run_dir = os.path.join(RUNS_DIR, parts[i + 1])
                if os.path.exists(os.path.join(run_dir, "run.json")):
                    return run_dir
    return None


def record_eval(run_dir, dataset, metrics, settings=None, checkpoint=None):
    """Append one evaluation to <run_dir>/results.json. Returns the entry, or None if no run_dir."""
    if not run_dir:
        return None
    path = os.path.join(run_dir, "results.json")
    entries = []
    if os.path.exists(path):
        with open(path) as f:
            entries = json.load(f)
    entry = {
        "dataset": dataset,
        "time": utc_now(),
        "checkpoint": checkpoint,
        "metrics": metrics,
        "settings": settings or {},
    }
    entries.append(entry)
    _dump(path, entries)
    print(f"[runlog] recorded {dataset} eval → {path}")
    return entry
