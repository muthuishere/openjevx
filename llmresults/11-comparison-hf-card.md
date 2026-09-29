## How it compares

Measured on 2026-09-29. Every model got the same questions, one question per request, 100 questions per set
(so each number is within about ±9 points). "Confident" means the model gave its answer at least 80%.
Jeff-0.8B ([mstrasser/Jeff-Qwen3.5-0.8B](https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B)) is the closest
open model in size and API; it ran on Apple silicon (MLX), OpenJevX on CPU.

**Everyday rules and logs** (the work OpenJevX is built for)

| Set | OpenJevX: confident and right | OpenJevX: confident and wrong | Jeff-0.8B: confident and right | Jeff-0.8B: confident and wrong |
|---|---|---|---|---|
| Everyday rules | 100% | 0% | 86% | 0% |
| Rules with conditions | 91% | 5% | 82% | 0% |
| Log triage | **93%** | 7% | 38% | 8% |
| Real public postmortems | 50% | **18%** | 71% | 8% |
| All four | **83.5%** | 7.5% | 69.2% | 4.0% |

**General benchmarks** (Jeff's public panel: BBH, Financial PhraseBank, JudgeBench, RAGTruth, WinoGrande)

| | OpenJevX | Jeff-0.8B |
|---|---|---|
| Accuracy | 60.4% | **75.4%** |
| Confident on | 11% | 65% |
| Confident and wrong | 1.0% | 9.6% |

**Size and speed**

| | OpenJevX | Jeff-0.8B |
|---|---|---|
| Model | 421M, 8-bit ONNX, 598 MB | 0.8B, 16-bit, 1.7 GB |
| Runs on | any CPU, one binary, no Python | Python + PyTorch, or MLX on Apple silicon |
| Median time per question | 450 ms (CPU, shared with another job) | 34 ms (Apple GPU) |

**Where it stands**

- Use it for rule-shaped decisions and log triage: it commits to an answer on 91% of those and is right 92% of
  the time when it does.
- It is not a general reasoner. On the public benchmarks it is mostly unsure (confident on 11%), so it rarely
  misleads there but rarely helps either. Larger models, and hosted Jev, do much better on reasoning.
- Known miss: on real incident postmortems it is confidently wrong 18% of the time. Check those answers.
- The rules set shares facts with the training data (the questions are reworded), so its 100% overstates
  performance on new rules.
