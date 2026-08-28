"""explog — views over the run records in experiments/runs/.

Standard library only, so it runs anywhere the repo is checked out (laptop, server, CI).

    python -m explog table            regenerate experiments/INDEX.md (the glance)
    python -m explog show <id>        one run in full
    python -m explog compare <a> <b>  config diff + metrics side by side
    python -m explog note <id> ...    write validity / reason / verdict / idea into notes.md
    python -m explog brain-sync       emit ledger files for the research-buddy brain
    python -m explog check            hygiene report
"""
