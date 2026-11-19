# Diagnosing Reasoning Loops in Qwen3.5-0.8B

*Testing published weight surgery on Swedish arithmetic*

**Abdolamir Karbalaie · SE.LLMA candidate research note · 19 November 2026**

## Research question

Does applying the published p50c weight edit to Qwen3.5-0.8B increase the proportion of correct, naturally completed answers on held-out Swedish arithmetic compared with the original model with thinking enabled? How does it compare with the original model with thinking disabled?

**Hypothesis.** If the edit removes a failure that blocks the model from answering, it should reduce repetition and increase correct, naturally completed answers relative to the original model with thinking enabled. Fewer loops without a gain in successful answers would challenge its practical benefit.

## Answer to the research question

With thinking enabled, p50c increased correct, naturally completed Swedish answers from **19/40 to 29/40**. The paired difference was **+25 percentage points** (95% family-bootstrap interval **0 to 50**). Without the edit, turning thinking off scored **31/40** and averaged **22 output tokens**, versus **2,463** for p50c with thinking on. The B–C difference was −5 points (95% interval −25 to 15). Thinking off is the practical baseline for short, structured tasks like these.

## Source critique

Takehi and colleagues used GPT-6 Astra to inspect weights and select edits using repeated practice feedback. The p50c search took 158 turns, 80 Python runs and 50 forward-pass calls. It changed 130,188 BF16 values in two feed-forward matrices. The editor reported that 576 of 579 practice rescues came from previously truncated outputs. The separate 10, 25 and 50-call runs produced different edits, and their benchmark table lacks a random-edit comparison. A follow-up on Gemma 3 270M, which rarely looped, found no established math gain. This study tests transfer of the published repair and diagnoses failures from observed outputs. [Original study](https://rikiyat.github.io/blog/weight-space-programming/) · [Experiment logs](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/README.md) · [p50c editor report](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/final_answer.md)

## Experimental design

I used the pinned Qwen3.5-0.8B release and reconstructed the p50c patch against the authors' tensor hashes. Forty arithmetic concepts from 20 wording families each had Swedish and English versions with a common gold answer. Ten separate bilingual development concepts informed feasibility and the token cap. Sixteen short instruction and comprehension prompts screened for damage. The test configuration was frozen before generation and all **672 planned requests were saved**.

The primary endpoint required a natural stop within **8,192 output tokens** and one valid final answer matching the gold value. Prompts asked for a separate `Svar:` or `Answer:` line. Runs shared a Colab NVIDIA L4, vLLM 0.33.0, BF16, decoding seed 17 and the same per-item seeds. The primary contrast was B–A on Swedish arithmetic. The 95% intervals use 10,000 resamples of wording families. R1 randomizes patch signs. S1 uses a similarly sized, approximately orthogonal rank-one direction in the same columns.

## Held-out results

**Table 1. Held-out Swedish and English arithmetic by condition**

| Condition | Swedish successes / 40 | English successes / 40 | Swedish cap hits / 40 | Mean Swedish output tokens |
| --- | ---: | ---: | ---: | ---: |
| A · Original, thinking on | 19 | 37 | 17 | 5,113 |
| B · p50c, thinking on | 29 | 40 | 0 | 2,463 |
| C · Original, thinking off | 31 | 40 | 0 | 22 |
| C_pen · Original, thinking on, repetition penalty 1.08 | 19 | 39 | 0 | 2,552 |
| D · p50c, thinking off | 31 | 40 | 0 | 20 |
| R1 · Random-sign edit, thinking on | 7 | 25 | 30 | 6,817 |
| S1 · Orthogonal edit, thinking on | 18 | 30 | 17 | 4,875 |

Each condition used Swedish and English versions of the same 40 held-out arithmetic concepts. Success required a correct `Svar:` or `Answer:` line and a natural stop within 8,192 output tokens. A cap hit reached that limit. Mean tokens include reasoning and final text. Source [condition summary](results/test-370e9d84ae06/summary.csv) (test rows).

With thinking on, p50c (B) increased Swedish successes from 19/40 to 29/40 and English successes from 37/40 to 40/40 (Table 1). For Swedish, the gain was 25 percentage points (95% interval 0 to 50, [Figure 3](REPORT.pdf#page=6)). Swedish cap hits fell from 17 to zero ([Figure 1](REPORT.pdf#page=5)), and outputs flagged for repetition fell from 19 to zero.

The original model with thinking off (C) scored 31/40 in Swedish and averaged 22 output tokens, versus 2,463 for B (Table 1). B was 5 percentage points below C (95% interval −25 to 15, [Figure 3](REPORT.pdf#page=6)). C_pen eliminated Swedish cap hits and flagged repetition, while its score stayed at A’s 19/40. The two single-draw control edits scored 7/40 (R1) and 18/40 (S1).

### What changed on individual questions

B succeeded on 15 Swedish questions that A missed and missed five that A got right, for a net gain of 10 ([Figure 2](REPORT.pdf#page=5)). Twelve of those 15 rescues had hit the cap and been flagged for repetition under A. On the book question ([Appendix D, Addition 0-v0 SV](REPORT.pdf#page=8)), A hit 8,192 tokens and B stopped after 2,004 tokens with the correct answer, 22.

The post hoc review found the correct number in all 11 Swedish final texts that missed the endpoint under B. On the pencil question ([Appendix D, Subtraction 0-v0 SV](REPORT.pdf#page=9)), A wrote `Svar: 13` and B wrote `Svars: 13`. The extra letter made B’s answer fail the format rule. Under the [exploratory numeric-only rule](results/test-370e9d84ae06/numeric_only_posthoc.csv), B scored 40/40 in Swedish, A 23/40, C 34/40, and C_pen 38/40.

D and C got the same five of eight probes right in each language ([Appendix E](REPORT.pdf#page=13)). The language detector agreed with 23 of 40 Claude-assisted reference labels, including 4 of 20 English labels ([Figure 5](REPORT.pdf#page=7)).

In `results/test-370e9d84ae06/`, `paired_effects.csv` gives paired counts and intervals, `summary.csv` gives repetition rates, and `rescues.json` flags capped and repetitive rescues. In `predictions_compact.jsonl`, look up `test-addition-0-v0` and `test-subtraction-0-v0` (language `sv`) for the raw outputs.

## Scope and limitations

The four-hour assignment target shaped the scope. I evaluated one existing 0.8B checkpoint and a published edit on **40 held-out Swedish questions from 20 wording families**, with English versions of the same 40 concepts. The runs used one decoding seed. I used no large training dataset and performed no new training or fine-tuning. The sample limits precision, the primary interval includes zero, and the 16 probes offer only a small check for harm. The synthetic tasks and two sampled control edits limit broader conclusions.

## Recommendation and reproduction

I recommend a small follow-up on Swedish tasks that benefit from thinking. For short arithmetic, I would start with the original model with thinking disabled. Use fresh wording families and several decoding seeds. Before generation, define a shared answer format and a blinded rubric for numeric correctness. Report completion, numeric accuracy, and format compliance separately. Add ordinary Swedish LoRA fine-tuning if those results identify a gap worth addressing.

LLM-JEPA adds a loss that trains the model to predict an internal representation alongside the usual text prediction loss. If the simpler comparisons leave a gap, a follow-up could compare the original model, p50c, ordinary LoRA and LoRA with the extra loss. Match the training data and settings across the two LoRA conditions. The published Qwen3-1.7B experiment on English GSM8K reported a change from 44.32% to 45.00%. This study ran neither training condition. [LLM-JEPA, Table 5](https://arxiv.org/html/2509.14252v2#S5.SS1)

The [frozen configuration](results/test-370e9d84ae06/frozen.json) and [weight manifest](results/test-370e9d84ae06/weights_manifest.json) record the source fingerprints, model and patch revisions, and seed. The [repository guide](README.md) gives CPU reanalysis and Colab rerun steps.

See the [full report](REPORT.pdf) and [supplementary results](results/supplementary/).
