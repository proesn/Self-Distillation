validity: pending
reason: 
verdict: 
idea: 
---
Baseline SDFT on kkp, A6000 profile (5120/4096, vLLM 0.4, grad ckpt), 1 epoch = 309 steps, 16 h 11 min (~189 s/step), peak 43.1 GB, code cf15508 (tracked tree clean, env.sh untracked). In-training eval (first 100 eval puzzles, greedy, 4096 tok) every 30 steps: 57 57 59 51 59 57 51 53 46 50 59 — flat within ±5 pt noise. Loss 0.064 → 0.036, kl_approx 0.080 → 0.044, completions 1813 → 1009 tokens, time/generate 101 → 48 s: style/length convergence without accuracy gain (matches the advisor's 'OPSD weak on K&K'). Pending: standalone eval of checkpoint-300 + base on all 300 puzzles (launch 2026-08-30_eval-kkp-baseline) and the teacher view (2026-08-31_eval-teacher-view-base) — whether base + demonstration beats base + plain prompt decides if the flat curve is missing signal or a training problem. Record: code.patch, launch.sh, launcher.sh and notes.md were not in the pushed record; regenerated 2026-08-31 from run.json (launcher.sh byte-exact, sha256 match; code.patch empty by inference — same dirty-only-via-env.sh state as calib2).
