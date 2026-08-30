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
    modified = [line[3:] for line in porcelain.splitlines() if line.strip() and not line.startswith("??")]
    return {
        "sha": _git("rev-parse", "HEAD").strip() or None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD").strip() or None,
        "dirty": bool(porcelain.strip()),
        "dirty_tracked": bool(modified),  # tracked files changed → code.patch is non-empty and matters
        "modified": modified,
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


class GpuMemorySampler:
    """Background thread sampling device-level used memory (total - free) via cudaMemGetInfo.

    torch's allocator statistics are misleading with a colocated vLLM engine in sleep mode (virtual
    reservations stay counted after the physical memory is released), so the device is asked directly.
    `peak_and_reset()` returns the max used GB since the previous call.
    """

    def __init__(self, interval=0.5):
        import threading

        self.interval = interval
        self._peak = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="gpu-mem-sampler", daemon=True)

    def start(self):
        try:
            import torch

            if not torch.cuda.is_available():
                return self
        except Exception:
            return self
        self._thread.start()
        return self

    def _run(self):
        import torch

        while not self._stop.is_set():
            try:
                free, total = torch.cuda.mem_get_info()
                used = (total - free) / 2**30
                with self._lock:
                    self._peak = max(self._peak, used)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def peak_and_reset(self):
        with self._lock:
            peak, self._peak = self._peak, 0.0
        return round(peak, 2) if peak else None

    def stop(self):
        self._stop.set()


def instrument_timing(trainer):
    """Time the three phases of a step into trainer._metrics so they reach events.jsonl / wandb.

    time/generate  — sampling + tokenising + IS forward (once per optimizer step)
    time/loss      — one student fwd/bwd + teacher fwd (once per sequence; logged as the mean)
    time/vllm_sync — weight push into the engine (once per optimizer step)
    """
    import functools
    import time

    def timed(name, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                mode = "train" if trainer.model.training else "eval"
                trainer._metrics[mode][name].append(time.perf_counter() - t0)

        return wrapper

    trainer._generate_and_score_completions = timed("time/generate", trainer._generate_and_score_completions)
    trainer._compute_loss = timed("time/loss", trainer._compute_loss)
    trainer._move_model_to_vllm = timed("time/vllm_sync", trainer._move_model_to_vllm)
    return trainer


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
    """Row count + sha256 over 64 evenly spaced rows — identifies the data content, not the loader code."""
    n = len(dataset)
    h = hashlib.sha256()
    for i in sorted({int(k * (n - 1) / 63) for k in range(64)}) if n else []:
        h.update(json.dumps(dataset[i], sort_keys=True, default=str).encode())
    return {"rows": n, "fingerprint": f"sha256:{h.hexdigest()[:16]}" if n else None}


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
            "kind": "train",
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
        self.sampler = GpuMemorySampler().start()
        self.peak_seen = None

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
        peak = self.sampler.peak_and_reset()
        if peak is not None:
            row["gpu_used_peak_gb"] = peak
            self.peak_seen = max(self.peak_seen or 0.0, peak)
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def on_train_end(self, args, state, control, **kwargs):
        self.sampler.stop()
        if self.peak_seen is not None:
            self.record.set(peak_gpu_mem_gb=self.peak_seen)
        self.record.finalize("finished")


# ---------- standalone evaluations: records of their own ----------

def _checkpoint_step(path):
    m = re.search(r"checkpoint-(\d+)", path or "")
    return int(m.group(1)) if m else None


class EvalRecord(RunRecord):
    """One standalone evaluation = one record folder (`kind: eval`), linked to the training run it targets.

        experiments/runs/<YYYY-MM-DD_label>/
            run.json      kind eval · target {run, model, adapter_path, checkpoint_step} · dataset · settings · git · launch · status
            results.json  [{dataset, metrics, per_sample, responses}]   (responses stay outside git; the path is recorded)
            launch.sh / launcher.sh / code.patch / notes.md            as for training runs
    On finish, a pointer entry {dataset, metrics, eval_run} is appended to the target training run's results.json.
    """

    @classmethod
    def start(cls, label, dataset, model, adapter_path=None, target_run=None, settings=None, num_samples=None):
        if target_run is None:
            d = infer_run_dir(adapter_path, model)
            target_run = os.path.basename(d) if d else None
        rec = cls(make_run_id(label))
        os.makedirs(rec.dir, exist_ok=True)
        git = git_state()
        with open(os.path.join(rec.dir, "code.patch"), "w") as f:
            f.write(_git("diff", "HEAD"))
        launch = launch_info()
        write_launch_script(rec.dir, launch, git)
        if launch["launcher"] and os.path.exists(launch["launcher"]):
            shutil.copyfile(launch["launcher"], os.path.join(rec.dir, "launcher.sh"))
        rec.data = {
            "schema_version": SCHEMA_VERSION,
            "id": rec.run_id,
            "kind": "eval",
            "name": label,
            "group": None,
            "tags": [],
            "idea": [],
            "status": "running",
            "started": utc_now(),
            "ended": None,
            "host": socket.gethostname(),
            "gpu": gpu_name(),
            "git": git,
            "cmd": shlex.join(sys.argv),
            "launch": launch,
            "target": {"run": target_run, "model": model, "adapter_path": adapter_path, "checkpoint_step": _checkpoint_step(adapter_path)},
            "data": {"name": dataset, "num_samples": num_samples},
            "settings": settings or {},
            "env": package_versions(),
            "metrics": None,
            "error": None,
        }
        rec.save()
        atexit.register(rec._atexit)
        rec.write_notes_skeleton()
        print(f"[runlog] eval record {rec.dir} (target run: {target_run or 'none — base model'})")
        return rec

    def finish(self, metrics, per_sample=None, responses_path=None):
        entry = {
            "dataset": self.data["data"]["name"],
            "time": utc_now(),
            "checkpoint": self.data["target"]["adapter_path"] or self.data["target"]["model"],
            "metrics": metrics,
            "settings": self.data["settings"],
            "per_sample": per_sample,
            "responses": responses_path,
        }
        _dump(os.path.join(self.dir, "results.json"), [entry])
        self.data["metrics"] = metrics
        self.finalize("finished")
        target = self.data["target"]["run"]
        if target:
            record_eval(os.path.join(RUNS_DIR, target), entry["dataset"], metrics, settings=entry["settings"], checkpoint=entry["checkpoint"], eval_run=self.run_id)


# ---------- results.json pointers on the training run ----------

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


def record_eval(run_dir, dataset, metrics, settings=None, checkpoint=None, eval_run=None):
    """Append one evaluation to <run_dir>/results.json (a pointer when it came from an EvalRecord). Returns the entry, or None if no run_dir."""
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
        "eval_run": eval_run,
    }
    entries.append(entry)
    _dump(path, entries)
    print(f"[runlog] recorded {dataset} eval → {path}")
    return entry
