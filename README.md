Swissdox – Media Analysis Pipelines

This repository contains the data collection, processing, and analysis pipelines used in a research project on media discourse about public administration in Switzerland.
The project relies on articles retrieved from the Swissdox media database and applies two alternative NLP pipelines, complemented by a systematic model comparison and evaluation framework.

Project overview
The repository implements three main components.
Data collection
Retrieval and cleaning of Swiss media articles via the Swissdox API.

Analysis pipelines

Embedded + RoBERTa pipeline: classical NLP approach combining multilingual embeddings for topic classification and XLM-RoBERTa for sentence-level sentiment analysis.

LLM (Apertus) pipeline: article-level classification using a locally hosted OpenAI-compatible LLM server.

Evaluation and comparison

Alignment and comparison of outputs from the LLM pipeline, the embedded + RoBERTa pipeline, and an external GPT-based annotation pipeline.

Repository structure

SWISSDOX_REPO
├── src
│ └── swissdox
│ ├── api.py
│ ├── embeddings.py
│ ├── sentences.py
│ ├── sentiment.py
│ ├── llm_client.py
│ ├── pipelines
│ │ ├── emb_roberta.py
│ │ └── llm_apertus.py
│ └── evaluation
│ ├── compare.py
│ └── plots.py
├── scripts
│ ├── download_articles.py
│ ├── run_emb_roberta.py
│ ├── run_llm_apertus.py
│ ├── run_llm_apertus_overnight.py
│ ├── run_comparison.py
│ └── run_comparison_plots.py
├── data
│ ├── raw
│ ├── processed
│ └── external
├── notebooks
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md

Environment setup
Create and activate a virtual environment using Python 3.12, then install dependencies from requirements.txt and install the project in editable mode.
Create a .env file at the repository root containing the Swissdox API credentials and optionally a HuggingFace token.

Data collection
Swiss media articles are retrieved and cleaned using the Swissdox API.
Running the download script produces a raw parquet file containing the articles.
Output location: data/raw/swissdox_articles_raw.parquet

Pipeline A – Embedded + RoBERTa
This pipeline assigns article-level topics using multilingual embeddings, splits articles into sentences, filters sentences based on administrative keywords, and applies XLM-RoBERTa sentiment analysis at sentence level.
Running the pipeline produces cleaned and enriched parquet files and a CSV export with sentence-level annotations.
Outputs are stored in data/processed.

Pipeline B – LLM (Apertus)
This pipeline performs article-level classification using a locally hosted OpenAI-compatible LLM server.
It produces topic classification, sentiment toward public administration with justification, and populism detection with justification.
Outputs are stored as parquet files in data/processed.
An overnight runner is available for long or resource-constrained runs and supports automatic restarts and checkpointing.

Evaluation and comparison
The evaluation framework aligns outputs from three annotation sources: the LLM pipeline, the embedded + RoBERTa pipeline, and an external GPT-based annotation pipeline.
Input files can be provided in CSV or Parquet format and are not tracked in Git.
The comparison produces agreement metrics, confusion matrices, aligned datasets, and publication-ready figures.

lots (time series & composition)
This project includes a plotting suite to visualize monthly dynamics and model outputs consistently across techniques.
What it generates
For each technique (A, B, C):
Sentiment shares over time (monthly)
Line plot showing the monthly share of POSITIVE / NEGATIVE / NEUTRAL.
Topic shares over time (monthly, stacked)
Stacked bar chart normalized to 1, showing how topic composition evolves over time.
Negativity + department composition (scaled, single plot)
One combined plot where:
the line is the monthly negativity rate (NEGATIVE / total),
the stacked bars represent the composition of “negative” items by department proxy, scaled so that the total bar height equals the negativity rate.
Additionally (A and B only, if populism is available):
Populism YES share over time (monthly)
Line plot of the monthly share of populism == YES.
Input format
The plotting script accepts CSV or Parquet for each technique and standardizes labels internally.
Expected fields are standardized through rename maps (see scripts/run_comparison_plots.py).
Technique A is article-level (deduplicated by article id).
Techniques B and C are treated as sentence-level for plots (no deduplication).
Run
Example:
python scripts/run_comparison_plots.py \
  --a data/processed/swissdox_articles_labeled_llm.parquet \
  --b data/external/2025_GPT_ANALYSIS.csv \
  --c data/processed/swissdox_sentences_with_sentiment.parquet \
  --outdir outputs/plots
To save plots only (no display):
python scripts/run_comparison_plots.py \
  --a data/processed/swissdox_articles_labeled_llm.parquet \
  --b data/external/2025_GPT_ANALYSIS.csv \
  --c data/processed/swissdox_sentences_with_sentiment.parquet \
  --outdir outputs/plots \
  --no-show
Outputs
PNG figures are saved to the output directory (default: outputs/plots/), for example:
APERTUS_sentiment_shares.png
APERTUS_topic_shares_stacked.png
APERTUS_negativity_dept_scaled.png
APERTUS_populism_yes_share.png (if available)
(and analogous files for the other techniques).

Design principles
The repository follows a strict separation of concerns between reusable logic, executable scripts, and exploratory notebooks.
Pipelines are deterministic where possible, fully reproducible, and resumable.
All models share a harmonized label space to enable direct comparison.

Notes
Large data files are excluded from version control.
The LLM pipeline assumes a locally running OpenAI-compatible server.
Parquet is used internally for performance and robustness.

License and usage
This repository is intended for academic research purposes.