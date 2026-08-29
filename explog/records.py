"""Load run records and derive per-run summaries."""

import json
import os
from dataclasses import dataclass, field

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.environ.get("SDFT_RUNS_DIR", os.path.join(REPO_ROOT, "experiments", "runs"))

NOTE_FIELDS = ("validity", "reason", "verdict", "idea")
VALIDITIES = ("pending", "valid", "suspect", "invalid")


@dataclass
class Run:
    id: str
    dir: str
    meta: dict
    events: list = field(default_factory=list)
    results: list = field(default_factory=list)
    notes: dict = field(default_factory=dict)  # header fields + "body"

    # ---- convenience accessors ----
    @property
    def status(self):
        return self.meta.get("status", "?")

    @property
    def validity(self):
        return self.notes.get("validity") or "pending"

    @property
    def dataset(self):
        return (self.meta.get("data") or {}).get("name") or (self.meta.get("args") or {}).get("dataset_name") or "?"

    @property
    def model(self):
        return (self.meta.get("model") or {}).get("name") or "?"

    @property
    def cfg(self):
        return self.meta.get("config") or {}

    @property
    def args(self):
        return self.meta.get("args") or {}

    def eval_curve(self, dataset=None):
        """[(step, accuracy)] for eval/<dataset>_accuracy (default: the training dataset)."""
        key = f"eval/{dataset or self.dataset}_accuracy"
        return [(e["step"], e[key]) for e in self.events if key in e and e[key] is not None]

    def eval_datasets(self):
        names = set()
        for e in self.events:
            for k in e:
                if k.startswith("eval/") and k.endswith("_accuracy"):
                    names.add(k[len("eval/") : -len("_accuracy")])
        return sorted(names)

    def summary(self, dataset=None):
        curve = self.eval_curve(dataset)
        out = {"acc0": None, "best": None, "best_step": None, "final": None, "final_step": None, "delta": None}
        if curve:
            out["acc0"] = curve[0][1] if curve[0][0] == 0 else None
            best_step, best = max(curve, key=lambda t: t[1])
            out["best"], out["best_step"] = best, best_step
            out["final_step"], out["final"] = curve[-1]
            if out["acc0"] is not None:
                out["delta"] = out["final"] - out["acc0"]
        steps = [e["step"] for e in self.events if "loss" in e]
        out["last_step"] = max(steps) if steps else (max((e["step"] for e in self.events), default=None))
        losses = [e["loss"] for e in self.events if isinstance(e.get("loss"), (int, float))]
        out["final_loss"] = losses[-1] if losses else None
        peaks = [e["gpu_mem_peak_gb"] for e in self.events if isinstance(e.get("gpu_mem_peak_gb"), (int, float))]
        out["peak_gpu_mem_gb"] = max(peaks) if peaks else self.meta.get("peak_gpu_mem_gb")
        stamps = [e["time"] for e in self.events if "loss" in e and e.get("time")]
        out["sec_per_step"] = None
        if len(stamps) >= 3:
            from datetime import datetime

            t0, t1 = datetime.fromisoformat(stamps[0]), datetime.fromisoformat(stamps[-1])
            out["sec_per_step"] = round((t1 - t0).total_seconds() / (len(stamps) - 1))
        return out

    def standalone(self):
        """Latest standalone result per dataset from results.json → {dataset: metrics}."""
        out = {}
        for r in self.results:
            out[r["dataset"]] = r.get("metrics", {})
        return out


def _read_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _read_events(path):
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        pass
    return rows


def parse_notes(text):
    """Header `key: value` lines up to `---`, then free text as `body`."""
    fields = {"body": ""}
    if text is None:
        return fields
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == "---":
            i += 1
            break
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip()
            if k in NOTE_FIELDS:
                fields[k] = v.strip()
                i += 1
                continue
        break
    fields["body"] = "\n".join(lines[i:]).strip()
    return fields


def render_notes(fields):
    head = "".join(f"{k}: {fields.get(k, '') or ''}\n" for k in NOTE_FIELDS)
    body = (fields.get("body") or "").strip()
    return head + "---\n" + (body + "\n" if body else "")


def load_run(run_dir):
    meta = _read_json(os.path.join(run_dir, "run.json"), None)
    if meta is None:
        return None
    notes_path = os.path.join(run_dir, "notes.md")
    notes_text = None
    if os.path.exists(notes_path):
        with open(notes_path) as f:
            notes_text = f.read()
    return Run(
        id=meta.get("id") or os.path.basename(run_dir),
        dir=run_dir,
        meta=meta,
        events=_read_events(os.path.join(run_dir, "events.jsonl")),
        results=_read_json(os.path.join(run_dir, "results.json"), []),
        notes=parse_notes(notes_text),
    )


def load_runs(runs_dir=RUNS_DIR):
    runs = []
    if not os.path.isdir(runs_dir):
        return runs
    for name in sorted(os.listdir(runs_dir)):
        d = os.path.join(runs_dir, name)
        if os.path.isdir(d):
            r = load_run(d)
            if r is not None:
                runs.append(r)
    runs.sort(key=lambda r: r.meta.get("started") or "", reverse=True)
    return runs


def find_run(run_id, runs_dir=RUNS_DIR):
    """Exact id, else unique prefix / substring match."""
    runs = load_runs(runs_dir)
    for r in runs:
        if r.id == run_id:
            return r
    hits = [r for r in runs if run_id in r.id]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise SystemExit(f"no run matches {run_id!r}")
    raise SystemExit(f"{run_id!r} is ambiguous: " + ", ".join(r.id for r in hits))
