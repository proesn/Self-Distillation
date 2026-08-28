import argparse
import os
import sys

from . import brain, views
from .records import NOTE_FIELDS, VALIDITIES, find_run, load_runs, parse_notes, render_notes


def cmd_table(a):
    path, n = views.write_index()
    print(f"wrote {os.path.relpath(path)} ({n} runs)")
    if a.print:
        with open(path) as f:
            print(f.read())


def cmd_show(a):
    print(views.show(find_run(a.run_id)))


def cmd_compare(a):
    print(views.compare([find_run(r) for r in a.run_ids]))


def cmd_note(a):
    run = find_run(a.run_id)
    path = os.path.join(run.dir, "notes.md")
    fields = parse_notes(open(path).read()) if os.path.exists(path) else {"body": ""}
    for k in NOTE_FIELDS:
        v = getattr(a, k)
        if v is not None:
            fields[k] = v
    if a.append:
        fields["body"] = (fields.get("body", "") + "\n\n" + a.append).strip()
    if fields.get("validity") and fields["validity"] not in VALIDITIES:
        raise SystemExit(f"validity must be one of {VALIDITIES}")
    if fields.get("validity") == "invalid" and not fields.get("reason"):
        raise SystemExit("validity: invalid requires --reason")
    with open(path, "w") as f:
        f.write(render_notes(fields))
    print(f"updated {os.path.relpath(path)}")
    views.write_index()


def cmd_brain_sync(a):
    written, index = brain.sync(load_runs(), brain_experiments_dir=a.brain_dir, dry_run=a.dry_run)
    verb = "would write" if a.dry_run else "wrote"
    print(f"{verb} {len(written)} ledger files + {index}")
    for p in written:
        print("  ", p)


def cmd_check(a):
    problems = views.check(load_runs(), running_hours=a.running_hours)
    if not problems:
        print("ok — no problems")
        return
    for run_id, msg in problems:
        print(f"{run_id}: {msg}")
    sys.exit(1)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m explog", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("table", help="regenerate experiments/INDEX.md")
    s.add_argument("--print", action="store_true")
    s.set_defaults(fn=cmd_table)

    s = sub.add_parser("show", help="one run in full")
    s.add_argument("run_id")
    s.set_defaults(fn=cmd_show)

    s = sub.add_parser("compare", help="config diff + metrics side by side")
    s.add_argument("run_ids", nargs="+")
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("note", help="write judgement fields into notes.md")
    s.add_argument("run_id")
    for k in NOTE_FIELDS:
        s.add_argument(f"--{k}", default=None)
    s.add_argument("--append", default=None, help="paragraph appended to the notes body")
    s.set_defaults(fn=cmd_note)

    s = sub.add_parser("brain-sync", help="emit ledger files for the research-buddy brain")
    s.add_argument("--brain-dir", default=brain.DEFAULT_BRAIN_EXPERIMENTS, help="the brain's experiments/ folder")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_brain_sync)

    s = sub.add_parser("check", help="hygiene report (exit 1 if problems)")
    s.add_argument("--running-hours", type=float, default=24)
    s.set_defaults(fn=cmd_check)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
