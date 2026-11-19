# Reviewed experiment protocol v4

This protocol records the planned design. The frozen configuration in the results directory records the settings used for the completed run.

## Question and claim boundary

Does the published p50c patch improve correct, completed answers to held-out Swedish arithmetic questions within a declared token budget, relative to unchanged weights? Is it practically preferable to disabling thinking?

Reduced repetitive failures would be consistent with failure removal. Do not prespecify that this is the sole mechanism, claim newly learned Swedish capability, or claim independent discovery from weights alone.

## Model and intervention

Base: Qwen/Qwen3.5-0.8B at 2fc06364715b967f1860aea9cf38778875588b17.
Patch source: RikiyaT/lm_understand-logs at 4d09ba6d6a7e891dd8063974dfe81b3d78e3494f, exp8_math/runs/p50c/edit.

Reconstruct with the upstream patch scripts and preserve required notices. Verify original and edited tensor hashes, shapes and names. The patch affects 64 down_proj columns in each of layers 15 and 19; 130,188 BF16 entries actually change. Distinguish touched entries from changed entries.

Every condition begins from the same original checkpoint. Verify zero-edit and restoration exactly. Never layer randomized controls on top of the published patch. Preserve tokenizer, tied embeddings, generation defaults and text/vision key mappings. Use BF16 on supported hardware without quantization; unsupported precision requires a recorded protocol change and common dtype across all arms.

### Canonical artifacts and fallback boundary

Verify the canonical BF16 base/patch bytes on CPU before inference casting. On hardware without native BF16 support, FP16 is permitted only if this exact architecture and engine pass finite-logit, generation and memory smoke checks; convert every condition consistently and label the result a precision-modified evaluation. Record actual post-cast deltas as well as canonical hashes. A dtype change cannot be justified solely by assuming a free Colab tier supplies a particular GPU.

The source revision above is authoritative. Do not replace it with an unverified hash from review feedback. Use upstream apply_edit.py and st_numpy.py rather than inventing reconstruction logic. A prebuilt RikiyaT/Qwen3.5-0.8B-weight-space-programming download is an optional acquisition route: pin its immutable revision at download time, verify edited hashes and unchanged tensors against the canonical base, and use common tokenizer/template/generation settings. Downloading it has not yet been verified in this project.

If the compatible engine cannot run this exact model by the setup deadline, use a bounded same-model engine fallback or report a partial investigation. Do not silently replace it with Qwen3-0.6B/1.7B or a quantized GGUF: those change the intervention and question. A new model requires a separately designed study.

## Data and split

Target 10 development concept pairs, 40 held-out test concept pairs and 16 separate regression probes. Each pair has equivalent Swedish/English wording, one checked integer answer, shared concept_id and template_family_id. Prefer four balanced categories of simple one/two-step arithmetic.

Use original, deterministic examples. Avoid ambiguous wording, monetary rounding and locale-dependent conventions. Review all Swedish and English text and answers before test generation. Split by template family, not just numeric values; keep both languages together. Hash finalized datasets.

Test variants from the same family are not independent samples. Aim for at least 20 test families for 40 concepts, keep each family's variants together in uncertainty estimation, and report both counts. Development families must be disjoint. Do not claim a fixed minimum detectable effect without an explicit power calculation and assumptions.

The regression set contains 8 Swedish and 8 English comprehension, extraction or instruction-following probes with deterministic expected answers. These screen for obvious degradation, not preservation of general capability.

Development example only:
- Swedish: Maja har 12 böcker och får 7 till. Hur många böcker har hon nu? Avsluta med Svar: <heltal>.
- English: Maja has 12 books and receives 7 more. How many books does she have now? End with Answer: <integer>.
- Gold answer: 19.

Use the same content instruction across weight/mode conditions. Change thinking through the chat template, not an extra reasoning instruction. The development pilot may revise scorer and wording; test outputs may not guide either.

## Conditions

| ID | Weights | Thinking | Priority |
| --- | --- | --- | --- |
| A | Original | On | Required |
| B | Published p50c patch | On | Required |
| C | Original | Off | Required practical alternative |
| C_pen | Original; repetition_penalty=1.08 | On | Required thinking-on decoding alternative |
| D | Published p50c patch | Off | Required interaction/regression |
| R1 | Sign-randomized delta, seed 101 | On | Default core control |
| S1 | Random rank-one delta, seed 201 | On | Default core control |
| R2-R3 | Additional sign controls, seeds 102-103 | On | Optional, decided before testing |

R1 preserves the actual patch's nonzero support and entry magnitudes before casting: replace each nonzero delta with an independently sampled sign times its absolute value. This preserves Frobenius norm before rounding but disrupts direction and rank. Log actual support, norm and changed-entry count after BF16 rounding.

S1 uses the same two layers and selected 64 columns. In each layer let Delta be the actual patch delta restricted to those columns and v its unit leading right singular vector. Set g=Delta*v. Draw z from a standard Gaussian with the declared control seed, set u=z-g*(g^T*z)/(g^T*g), then normalize u. Guard against zero/near-zero g or u; any numerical resampling must be deterministic and logged. Set Delta_S1=||Delta||_F * u * v^T.

This has rank one and the same per-layer Frobenius norm, with zero Frobenius inner product against Delta before rounding up to numerical precision. It retains the leading column profile, not necessarily the exact rank/support of the rounded published patch. After casting, log actual norm ratio, changed entries, effective rank diagnostics and normalized inner product. Do not claim exact orthogonality after BF16/FP16 rounding.

A poor S1 result does not prove the target direction is uniquely causal: arbitrary perturbations can be destructive. Report regression damage and do not tune control strength on test outputs. A norm-titrated, quality-matched control is a separate feedback-calibrated follow-up.

One draw per control type gives limited evidence. Show each control separately. Do not select the most convenient random seed. Extra controls take priority over an optional second decoding seed if runtime permits.

## Engine, sampling and pilot

Use a Colab GPU; verify its name, available memory, BF16 support and software versions. Prefer vLLM when a compatible installation and smoke test succeed within 15 minutes of setup; otherwise use a tested Transformers fallback. Do not switch engines between conditions.

Target common sampling: temperature 0.6, top_p 0.95, top_k 20, presence_penalty 1.5, frequency_penalty 0.0 and repetition_penalty 1.0; decoding seed 17. C_pen changes only repetition_penalty to the fixed heuristic 1.08 while keeping thinking on and presence_penalty at 1.5. This tests one untuned sampler alternative, not an optimal policy. Verify and log whether the engine applies penalties to prompt tokens as well as generated tokens. An optional second seed 29 must be included for all conditions if enabled. Record engine seed and per-request seeds, prompt order, batch settings and scheduling configuration. A repetition penalty is not a substitute for presence penalty.

### Process, configuration and feasibility checks

Use sequential subprocesses per weight variant, waiting for workers to exit before loading the next variant. Group A, C and C_pen in the original-weight process; group B and D in the patched-weight process; run R1 and S1 in separate processes. This requires four weight variants for seven conditions. Do not mutate packed vLLM worker tensors via unverified internals. Measure loading, cleanup and generation time; no fixed per-load or total runtime is assumed.

Read the installed version's supported configuration. Start with unquantized weights and caches, a supported cache dtype such as auto with its resolved value recorded, and context length at least the longest rendered prompt plus output cap. Log attention KV and recurrent-state cache dtypes separately. Do not assume the literal CLI spelling bf16 works or claim a universal FP8 bug. Use language-model-only only if supported; it disables multimodal inputs and is not a tensor-name conversion. Choose batch size by measured memory headroom; do not hard-code 32 or 64 from model size alone.

Verify thinking on/off from rendered prompt token IDs, template kwargs, returned delimiters and finish metadata. A generated trace may start after a think-open token already in the prompt. Seeing prose is insufficient evidence that thinking mode was enabled.

Use a staged pilot: first short smoke generations to establish loading and template behavior; then four fixed development concept pairs across A/B with a candidate long cap for timing and censoring. Use the remaining development examples and representative controls as time permits. Benchmark either vLLM or its fallback adequately; do not spend the setup window on a broad engine competition.

On two fixed development prompts (one Swedish, one English), audit shared-trajectory scoring against shorter-cap requests at 1,024 and 4,096 for A/B when those budgets fit. Fix per-request seeds and batch settings. Investigate differences; matching once does not guarantee universal bitwise reproducibility. If independent reruns differ, preserve the valid saved-trajectory interpretation and document the limitation.

Run the development pilot with a candidate cap of 16,384 if feasible, otherwise 8,192. Inspect timing, natural completion and length distributions, not test outputs. Verify the thinking template, final delimiter, EOS behavior, patch restoration, scorer and language annotations. No forced EOS, budget-aware reasoning prompt/controller or cap-dependent logits processing may be used for shared-trajectory budget curves.

Choose the largest candidate cap among 16,384 and 8,192 whose projected complete core fits the 85-minute generation slot with a 20% timing reserve. Prefer retaining a useful long cap while reducing sample count before shortening all runs. Use median/worst-case pilot timings across original, edited and a random arm, plus loading overhead; do not extrapolate from the fast no-thinking arm alone.

Evaluate candidate scope in this order: 40 test pairs with seven conditions and one seed; 24 pairs with the same conditions; then 16 pairs with the same conditions. Choose subsets deterministically and balance categories before test generation. Never shorten the primary cap below 8,192. If all seven arms at 16 pairs cannot fit, predeclare a five-arm A/B/C/D/C_pen pilot without R1/S1; report that specificity controls are absent. If that cannot fit, stop at the development pilot.

An increase to 60 or 80 pairs is optional only if the development timing projection, data-review effort and additional independent families fit the same schedule with reserve. Decide before opening test outputs. Do not promise a fixed MDE or a fixed 15-20 minute cost. Optional extra sign-control draws and a second decoding seed also require a fully budgeted, pre-test decision; never add them because the primary result is nearly significant.

At 40 pairs plus 16 probes, seven conditions and one seed, there are 672 generations: at most 5,505,024 output tokens at an 8,192 cap or 11,010,048 at 16,384. This excludes development and optional perplexity and is not a runtime guarantee.

Freeze one primary budget equal to the selected maximum generation cap. Record why it and the sample size were chosen based on timing and censoring, not the size of the edited advantage. All selected conditions use this cap and the same test IDs. If neither A nor B completes enough development tasks to measure correctness meaningfully, increase the cap within budget or label the study a budget-limited pilot.

Predeclare execution groups in the order original (A, C, C_pen), patched (B, D), R1, S1 with a fixed prompt ordering. Under an unexpected interruption, retain all outputs, report missing condition/ID counts and distinguish intended from completed scope. Primary paired analysis uses the common A/B IDs with missingness disclosed; do not compare unmatched denominators or silently drop missing controls. No result-dependent condition removal is allowed.

## Budget curves and scoring

Save raw generated token IDs and text, including thinking delimiters. For each shared trajectory, evaluate budgets 512, 1,024, 2,048, 4,096, 8,192 and 16,384 only up to the selected cap.

Primary endpoint: a naturally completed response by the primary budget with a correct, unambiguous final answer. Requiring natural EOS/completion is explicit; a cap-hit response with an answer marker is recorded as answer-present but does not satisfy this completed-response endpoint.

For every budget:
- Count natural completion by that budget.
- Count correct natural completion by that budget.
- Record whether a valid complete answer marker is already present, separately.
- Keep incomplete/cap-hit outputs in denominators as failures of the bounded endpoint.
- Retain cap-hit status as censoring for unbounded completion, not evidence of an infinite loop.

Parse final content after the generated thinking-close delimiter when thinking is enabled. For no-thinking templates that pre-close reasoning in the prompt, parse the generated final content directly. Validate this against actual tokenized templates. Never extract an arbitrary last number or grade a number from unfinished thinking. Normalize whitespace, handle documented answer labels, and reject incomplete numeric tokens or ambiguous markers. Add tests for those cases.

A single long sampled trajectory supplies prefix outcomes; separately rerunning with a shorter cap need not be bitwise identical because of engine scheduling or cap-dependent logic. Validate cap independence on development examples and document deviations. A short-prefix curve is not a measurement of short-run wall-clock latency.

With a common final administrative cap, plot direct completion proportions against tokens; no survival-analysis dependency is required. Do not extrapolate beyond the cap or infer eventual completion probability.

## Other measurements

Record valid-final-answer rate, natural completion, cap-hit rate, generated tokens, elapsed run time, regression accuracy and a repetition proxy. D-C with thinking off is the primary regression contrast. A/B probe scores remain descriptive because failure to finish can dominate them; no small probe set establishes general capability preservation.

Repetition proxy: 1 - distinct_token_4grams / total_token_4grams over generated content, excluding control tokens. Flag outputs with at least 64 content tokens and a fraction above 0.5; calibrate only on development data. It is imperfect for long mathematical prose. Inspect anonymized flags and report disagreements. Repetition and truncation are separate.

For Swedish prompts, annotate three views separately: first 250 generated reasoning tokens (or all if shorter), the full observable reasoning segment, and final prose. Use Swedish, English, mixed/other, or insufficient text. Mark repetition in the prefix as well; it is not guaranteed to precede looping or reflect productive reasoning. Numeric-only answers cannot establish language fidelity. Record truncated reasoning separately.

Use a fixed conservative sv/en stopword screen with mixed/other and insufficient labels, validated against 40 blind manual labels stratified across the three views. Unique stopword evidence limits domination by a repeated word. Do not assume reliability or tune the detector using test labels; report class-specific errors and restrict claims to the manual sample if validation is poor. Do not train a new Scandinavian classifier in the core. The views distinguish initial output language from language dominated by later repetition, but do not expose inaccessible internal reasoning. These measurements remain descriptive.

Optional perplexity: at most 100 deterministically selected aligned Swedish/English FLORES devtest sentence pairs, original and edited weights first. Verify authorized dataset access, revision, IDs and license. Score with a common tokenizer/BOS/context convention; mask padding and compute total token NLL divided by scored-token count, then exponentiate. Compare edits within each language, not raw Swedish against English perplexities. Keep this independent from prompt selection and omit it if setup threatens the core schedule. Existing regression probes remain mandatory.

Do not add output-attribution classifier training or a new steering/edit-discovery experiment to this run.

## Analysis and decision rules

Primary contrast: B-A on Swedish correct-completed-response rate at the declared primary budget. Everything else is secondary/exploratory: B-C, B-C_pen, C_pen-A, D-C, random controls, English effects and language-by-edit interaction.

Average decoding seeds within concept if more than one is used. Bootstrap paired template-family clusters, keeping all concepts, languages, conditions and seeds together; if every concept has its own independent family this becomes a concept bootstrap. Use 10,000 replicates, bootstrap seed 20261118. Report denominators, family counts, percentage-point effects and 95% intervals. Few clusters limit inference. Do not treat translations, seeds or many budget thresholds as independent studies or search the curve for significance.

Report the paired correct/incorrect transition table, discordant concept and family counts, and any degenerate bootstrap intervals. An all-zero resampled interval with no discordant families does not establish equivalence or absence of harm. Do not report a universal MDE from sample size alone.

Classify B's rescues over A by A's repetition, truncation and answer-error status. Report overlap: truncated and repetitive are not disjoint. Such associations are consistent with failure removal, not causal mediation proof. Scores among items both models finish are descriptive and subject to selection bias.

- B improves over A and random controls with fewer repetitive failures: evidence for targeted transfer under this setup.
- C or C_pen is comparably accurate using fewer tokens: favor inference configuration operationally; absence of a significant B-C difference alone is not equivalence.
- D worsens relative to C: report mode-dependent collateral effects.
- More completion without better correctness: stopping/formatting gain, not demonstrated answer-quality gain.
- Random controls perform similarly: weaker evidence for special targeting.
- Null or uncertain results: no demonstrated benefit within measured budgets; longer-budget or broader effects remain open.
- Regression checks show no clear difference: report the measured uncertainty; do not claim non-inferiority.

## Logging, artifacts and authorization

Each prediction records run/concept/family IDs, split, language, condition, decoding/control seeds, rendered prompt, token IDs, raw output, thinking/final parsing, gold and extracted answers, each budget's score, repetition, tokens, stop reason and elapsed time.

Manifest: code commit and dirty state, model/tokenizer/patch revisions, tensor hashes, data/prompt hashes, dependency versions, hardware/dtype, resolved sampling and scheduling, selected cap/scope, all evaluated counts and exceptions.

MLflow records configurations, scalar metrics, budget-curve tables and artifacts; ordinary JSONL/CSV remains the primary portable evidence. Keep SQLite on the runtime's local disk. Back up a consistent closed database and artifact store, and save predictions incrementally to authorized durable storage.

### Durability and resume

Use one append-only JSONL prediction file, flush/fsync after each completed microbatch, and copy the closed file to authorized durable storage through a temporary file with a hash check. A corrupt partial line must be preserved and repaired explicitly, not silently skipped. Stable record keys contain config hash, dataset hash, condition, concept/probe ID, language and decoding seed. Resume skips only completed keys under the same configuration; duplicate/conflicting records and changed hardware/software are rejected. Do not select the best of retries.

Keep live SQLite on Colab local disk. Use one MLflow run per condition per phase and retain run IDs on resume. At condition boundaries and session end, pause tracking writes and use SQLite's backup API, then copy finalized artifacts. JSONL remains authoritative if tracking fails. A CPU resume/conflict test and repeating the smoke run without duplicate output replace the deliberate GPU interruption drill. No shard catalog is required.

The current implementation uses one decoding seed and one draw of each random control. Additional seeds/draws and optional perplexity are not implemented in this first notebook; they need a pre-test protocol/code revision. Layer-specific random seeds are 101+i and 201+i for layer index i in manifest order, with the root seeds preserved.

### Hard time limits and report readiness

All preparation, downloads and implementation count toward actual effort. Maintain an effort ledger and AI-use log as work proceeds. Stop planned test generation by minute 170; do not launch optional runs after that. Freeze new analyses at minute 195. Reserve 195-225 for the note and PDF, 225-232 for AI-use and third-party attribution, and 232-240 for reproduction/package checks and an authorized upload. If work is incomplete, document it; do not manufacture results or move work outside the reported effort window.

LoRA/SFT and Danish/Norwegian transfer are follow-up studies. The assignment's conventional-fine-tuning example is not a mandatory extra arm. No prediction that LoRA will underperform is justified by this experiment.
