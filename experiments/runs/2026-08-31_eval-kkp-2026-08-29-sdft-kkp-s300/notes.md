validity: pending
reason: 
verdict: 
idea: 
---
checkpoint-300 of 2026-08-29_sdft-kkp through vLLM LoRARequest, same settings as the base eval: accuracy 50.7 % vs base 49.0 % (+1.7 pt; SE ≈ 2.9 pt at n=300 → within noise), parse rate 98.7 % vs 83.7 %. SDFT taught the output format, not the logic — consistent with the flat in-training curve (57 → 59 on the first 100) and the completion-length collapse (1813 → 1009 tokens).
