validity: pending
reason: 
verdict: 
idea: 
---
Base model on 300 seeded training items: teacher 54.7 % vs student 33.0 % (gap +21.7; teacher-only 65, student-only 0, both 99). Even with the golden ReAct response in context the base scores only 55 % under the strict action + JSON-input scorer (copy ceiling 96.7 %): format compliance, not tool choice, is the limit. Split gap to check: base 0-shot is 33 % on training prompts here vs 58.8 % on the 97 eval prompts (in-training step 0) — the two splits differ in difficulty or format (llm_cl README: the tool-use train set was regenerated); do not compare train-split and eval-split numbers directly.
