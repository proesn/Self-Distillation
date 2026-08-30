validity: pending
reason: 
verdict: 
idea: 
---
Baseline SDFT on science, A6000 profile (2048/2048), 2 epochs = 166 steps, 3 h 04 min (~66 s/step, time/generate 20 s), peak 43.9 GB, code 65e91f6 clean. In-training eval (first 100 of 507, greedy) every 30 steps: 36 36 55 52 58 55 — +19 pt, best 58 @120. Loss 0.58 → 0.11, completions 228 → 310 tokens. Reading: clear gain; the demonstration's format + answer transfer (paper: Qwen2.5-7B science 66 → 70). Pending: standalone eval on the full 507-item split, teacher view.
