# AI use and human review

## Study design

I used the Medley approach to brainstorm the design with OpenAI ChatGPT, Google Gemini, Anthropic Claude, GLM, and DeepSeek. I compared their suggestions and checked them against the assignment and the original study. Their feedback helped refine the development and test split, generation budget, comparison conditions, regression probes, and interpretation. The final design decisions were mine. I consulted Claude Opus 5.5 as identified in my notes. Exact model versions were unavailable for some of the other consultations.

## Code, debugging, and Colab

ChatGPT and OpenAI Codex helped build the Python runner, AI-assisted synthetic bilingual task templates, tests, and step-by-step Colab notebook. They also helped set up the GPU runtime and work through the notebook cells. Codex diagnosed the indexed checkpoint loading problem and checked the rebuilt weights against the published hashes. I consulted Claude to review and debug the implementation. Claude helped fix answer parsing, improve vLLM batching, and strengthen the Colab setup and CPU checks.

I wrote and edited parts of the code, reviewed the workflow and notebook cells step by step, and used additional AI reviews to check the implementation before running the evaluation in Google Colab. I reviewed the research question, choices, and interpretation. [Validation](VALIDATION.md) records the software checks and describes how the completed run was checked against the saved predictions.

## Analysis and reporting

Claude helped prepare the first findings report, supplementary tables, and reference labels for the blinded language audit. The audit measures agreement against those Claude-assisted judgments. Gemini provided feedback on report organization and clarity. Codex recomputed the saved analyses, reconciled the MLflow records with the predictions, inspected output-format failures, and helped edit the Markdown, PDF, and Word reports.

The published p50c weight edit came from Takehi and colleagues, who used an AI editor in the original study. I evaluated their existing edit on a new task set.
