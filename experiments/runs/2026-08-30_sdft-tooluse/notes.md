validity: pending
reason: 
verdict: 
idea: 
---
Baseline SDFT on tooluse, A6000 profile (2048/1024), 2 epochs = 252 steps, 4 h 51 min (~69 s/step, time/generate 17.5 s), peak 44.1 GB, code 387e5d2 (only experiments/INDEX.md modified at launch — regenerated index, no code change). In-training eval every 30 steps on all 97 eval items (n=97 = whole split, so a standalone eval adds nothing but the LoRARequest-vs-merged-weights check): 58.8 61.9 59.8 57.7 56.7 58.8 60.8 58.8 59.8 — flat within ±2 items. Loss 0.45 → 0.07, completions 88 → 78 tokens. Reading: no accuracy gain; base already 59 %; the scorer's copy ceiling on demonstrations is ~97 % (nested-JSON regex). Pending: teacher view.
