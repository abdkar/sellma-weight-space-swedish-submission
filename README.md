# Diagnosing Reasoning Loops in Qwen3.5-0.8B

This SE.LLMA candidate study tests whether the published p50c weight edit helps Qwen3.5-0.8B answer Swedish arithmetic. With thinking enabled, strict success rose from **19/40 to 29/40** (paired 95% interval for the gain, **0 to 50 percentage points**). The original model with thinking disabled scored **31/40** with far fewer output tokens.

## 1. Code

Reanalyze the archived Colab outputs on CPU with Python 3.11 or later:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python reproduce_saved_analysis.py --phase test --output reanalysis
```

Use `--phase pilot` for the development run. The [script](reproduce_saved_analysis.py) verifies the source and task fingerprints, then regenerates strict summaries, paired comparisons, token-budget curves, and the exploratory numeric-only table.

To run generation, upload the [guided notebook](notebooks/colab.ipynb) to Google Colab, select a GPU with native BF16 support, and upload the GitHub repository ZIP in Section 2. Follow the notebook in order and consult the [frozen configuration](results/test-370e9d84ae06/frozen.json) for the recorded choices. The [protocol](PROTOCOL.md) documents the design. See [third-party notices](THIRD_PARTY_NOTICES.md) for upstream code and weights.

## 2. Results

The saved Colab run used an NVIDIA L4, vLLM 0.33.0, BF16, an 8,192-output-token cap, and seed 17. It contains 60 development predictions and 672 held-out and regression predictions.

- [All 116 questions and saved outcomes](results/supplementary/supplementary_all_116_questions.csv) and [supplementary workbook](results/supplementary/supplementary_tables.xlsx)
- [Compact test predictions](results/test-370e9d84ae06/predictions_compact.jsonl), [strict summary](results/test-370e9d84ae06/summary.csv), [paired effects](results/test-370e9d84ae06/paired_effects.csv), and [exploratory numeric-only results](results/test-370e9d84ae06/numeric_only_posthoc.csv)
- [Frozen configuration](results/test-370e9d84ae06/frozen.json), [model and patch verification](results/test-370e9d84ae06/weights_manifest.json), and [environment](results/test-370e9d84ae06/environment.lock.txt)
- [Development pilot](results/pilot-fe4f8fab8ef7/) and [MLflow condition metrics](results/mlflow_condition_metrics.csv)

## 3. Research note

The [Markdown research note](RESEARCH_NOTE.md) presents the hypothesis, study design, findings, limitations, and recommendation.

## 4. PDF report

The [final PDF](REPORT.pdf) includes the full results, figures, and appendices. Its references to Table 2 point to the post hoc numeric-only column in Table 1.

## 5. AI usage

The [AI-use disclosure](AI_USAGE.md) records the tools, their roles, and the candidate’s contributions and review. The [effort log](EFFORT_LOG.md) records the timing that could be verified.
