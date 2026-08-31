validity: pending
reason: 
verdict: 
idea: 
---
Base model on 300 seeded training puzzles: teacher prompt (puzzle + its verified Qwen CoT + 'answer with a response of your own') 89.3 % vs plain prompt 53.7 % — gap +35.7 pt; teacher-only 115, student-only 8, both 153; parse 98.3 % vs 80.0 %, gold parse 100 %. The demonstration is a strong in-sample signal (it contains the assignment), yet 10.7 % of teacher responses are wrong with the answer in context — the model re-derives rather than copies. Read together with the s300 eval: the signal exists but none of it reached unseen puzzles → the teacher's edge is answer-conditioned, and a student without the answer cannot learn to emit the commit tokens. Not measured: out-of-sample 1-shot ICL (a different solved puzzle in context).
