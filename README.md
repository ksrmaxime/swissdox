Swissdox – Media Analysis Pipelines
Overview
This repository contains the data collection, processing, analysis, and evaluation pipelines developed for a research project on media discourse about public administration in Switzerland.
The project relies on articles retrieved from the Swissdox media database and implements multiple NLP-based annotation strategies, combined with a systematic model comparison and evaluation framework.
The repository is designed to support reproducible research, large-scale text analysis, and direct comparison between classical NLP approaches and LLM-based pipelines.
Scope of the project
The pipeline covers the full workflow from raw media data to publication-ready figures:
Collection and cleaning of Swiss media articles via the Swissdox API
Article- and sentence-level annotation using multiple NLP techniques
Systematic comparison between annotation sources
Consistent time-series and compositional visualisations
Three annotation strategies are implemented and aligned:
Embedded + RoBERTa pipeline (classical NLP)
LLM pipeline (Apertus) using a locally hosted OpenAI-compatible server
External GPT-based annotation pipeline (input only, not tracked)
Repository structure
SWISSDOX_REPO/
├── src/
│   └── swissdox/
│       ├── api.py
│       ├── embeddings.py
│       ├── sentences.py
│       ├── sentiment.py
│       ├── llm_client.py
│       ├── pipelines/
│       │   ├── emb_roberta.py
│       │   └── llm_apertus.py
│       └── evaluation/
│           ├── compare.py
│           └── plots.py
├── scripts/
│   ├── download_articles.py
│   ├── run_emb_roberta.py
│   ├── run_llm_apertus.py
│   ├── run_llm_apertus_overnight.py
│   ├── run_comparison.py
│   └── run_comparison_plots.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/        # exploratory only
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
Environment setup
Python version: 3.12
Create and activate a virtual environment
Install dependencies:
pip install -r requirements.txt
pip install -e .
Create a .env file at the repository root containing:
Swissdox API credentials (required)
HuggingFace token (optional, for some models)
Data collection
Swiss media articles are retrieved via the Swissdox API and cleaned automatically.
Output format: Parquet
Output location:
data/raw/swissdox_articles_raw.parquet
The cleaning step standardises text fields and metadata for downstream analysis.
Analysis pipelines
Pipeline A – Embedded + RoBERTa
A classical NLP pipeline combining multilingual embeddings and transformer-based sentiment analysis.
Main steps:
Article-level topic classification using multilingual embeddings
Sentence segmentation
Keyword-based sentence filtering (administrative focus)
Sentence-level sentiment analysis using XLM-RoBERTa
Outputs:
Cleaned and enriched Parquet files
Sentence-level CSV export
Stored in:
data/processed/
Pipeline B – LLM (Apertus)
An article-level annotation pipeline based on a locally hosted OpenAI-compatible LLM server.
For each article, the pipeline produces:
Topic classification
Sentiment toward public administration (with justification)
Populism detection (with justification)
Features:
Robust checkpointing
Automatic restart support
Dedicated overnight runner for long or resource-intensive jobs
Outputs:
data/processed/*.parquet
Evaluation and comparison
The evaluation framework aligns and compares annotations from:
Pipeline A (Embedded + RoBERTa)
Pipeline B (LLM – Apertus)
External GPT-based annotations (input only)
The framework produces:
Agreement metrics
Confusion matrices
Aligned datasets
Publication-ready figures
Input files may be provided in CSV or Parquet format and are not tracked in Git.
Plots and visualisations
A unified plotting suite ensures consistent visualisation across techniques.
What is generated (per technique)
Sentiment shares over time (monthly)
Line plot of POSITIVE / NEGATIVE / NEUTRAL shares
Topic shares over time (monthly, stacked)
Normalised stacked bars (sum = 1)
Negativity + department composition (scaled)
Combined plot where:
Line = monthly negativity rate
Stacked bars = department composition of negative items
Bar height scaled to negativity rate
Additionally (if available):
Populism YES share over time (monthly)
Input format and assumptions
CSV or Parquet accepted
Label harmonisation handled internally via rename maps
Technique A: article-level (deduplicated)
Techniques B & C: treated as sentence-level (no deduplication)
Example run
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
Design principles
Strict separation between:
reusable logic (src/)
executable scripts (scripts/)
exploratory notebooks (notebooks/)
Deterministic and resumable pipelines
Harmonised label spaces across models
Parquet used internally for robustness and performance
Notes
Large data files are excluded from version control
The LLM pipeline assumes a locally running OpenAI-compatible server
The repository is intended for academic research use
