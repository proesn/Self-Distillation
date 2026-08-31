validity: pending
reason: 
verdict: 
idea: 
---
Base model on 300 seeded training items: teacher 54.7 % vs student 33.0 % (gap +21.7; teacher-only 65, student-only 0, both 99). Even with the golden ReAct response in context the base scores only 55 % under the strict action + JSON-input scorer (copy ceiling 96.7 %): format compliance, not tool choice, is the limit. Split gap to check: base 0-shot is 33 % on training prompts here vs 58.8 % on the 97 eval prompts (in-training step 0) — the two splits differ in difficulty or format (llm_cl README: the tool-use train set was regenerated); do not compare train-split and eval-split numbers directly.

Split gap resolved (2026-08-31): train and eval share zero tools (407 vs 10 distinct, no overlapping instructions — cross-tool generalization by design), and the base 33 % vs 58.8 % difference is action-count composition, not prompt difficulty: by gold action count the base scores 1→0.52, 2→0.07, 3+→0.00 (teacher 0.74 / 0.29 / 0.17; train mean 1.82 actions/item vs eval 1.47). On single-action items base-on-train (0.52) ≈ base-on-eval (0.588). The eval split therefore tests mostly single-action unseen tools where the teacher's in-sample edge is smallest.
