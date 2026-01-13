# Swissdox – Media Analysis Pipelines

## Overview

This repository contains the data collection, processing, analysis, and evaluation pipelines developed for a research project on media discourse about public administration in Switzerland.

The project relies on articles retrieved from the Swissdox media database and implements multiple NLP-based annotation strategies. It includes a systematic framework for comparing a classical NLP approach against a modern LLM-based pipeline, from raw data to final visualizations.

## Key Features

-   **End-to-End Workflow:** A complete solution from raw data fetching via the Swissdox API to publication-ready figures.
-   **Comparative Analysis:** Implements and compares two distinct NLP approaches: a classical pipeline (multilingual embeddings + RoBERTa) and an LLM-based pipeline using a local model.
-   **Robust Processing:** Includes features like checkpointing and an overnight runner for resilient, large-scale data annotation with LLMs, managing memory and resuming automatically.
-   **Systematic Evaluation:** A dedicated framework to align, compare, and visualize results from different annotation sources, calculating agreement metrics and generating consistent plots.
-   **Modular & Reproducible:** A clean separation of core logic (`src/`) and executable scripts (`scripts/`), designed to support reproducible research.

## Repository Structure

```
swissdox/
├── .env.example
├── pyproject.toml
├── requirements.txt
├── README.md
├── scripts/                 # Executable scripts for the full workflow
│   ├── download_articles.py
│   ├── run_emb_roberta.py
│   ├── run_llm_apertus.py
│   ├── run_llm_apertus_overnight.py
│   ├── run_comparison.py
│   └── run_comparison_plots.py
└── src/                     # Core Python library
    └── swissdox/
        ├── api.py           # Client for the Swissdox API
        ├── llm_client.py    # Client for a local OpenAI-compatible LLM server
        ├── pipelines/       # End-to-end analysis pipelines
        │   ├── emb_roberta.py
        │   └── llm_apertus.py
        └── evaluation/      # Comparison and plotting logic
            ├── compare.py
            └── plots.py
```

## Getting Started

### Prerequisites

-   Python >= 3.12
-   Access to the Swissdox API.
-   (Optional) A locally running OpenAI-compatible server for the LLM pipeline.

### Installation

1.  **Clone and enter the repository:**
    ```bash
    git clone https://github.com/ksrmaxime/swissdox.git
    cd swissdox
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    # On Windows, use: venv\Scripts\activate
    ```

3.  **Install dependencies and the project package:**
    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```

4.  **Configure environment variables:**
    Create a `.env` file by copying the example:
    ```bash
    cp .env.example .env
    ```
    Then, edit the `.env` file with your credentials:
    -   `SWISSDOX_API_KEY`: Your key for the Swissdox API.
    -   `SWISSDOX_API_SECRET`: Your secret for the Swissdox API.
    -   `HF_TOKEN` (Optional): A Hugging Face token, required for downloading certain models used in the `Emb+RoBERTa` pipeline.

## Workflow

The project is structured as a sequence of executable scripts.

### Step 1: Download Articles

Fetch and clean media articles from the Swissdox API. The script uses predefined keywords and date ranges to query the database.

-   **Script:** `scripts/download_articles.py`
-   **Process:** Downloads data, cleans text and metadata, and saves the result.
-   **Output:** `data/raw/swissdox_articles_raw.parquet`

### Step 2: Run Analysis Pipelines

Choose one of the two implemented pipelines to annotate the downloaded articles.

#### A) The Embeddings + RoBERTa Pipeline (Classical NLP)

This pipeline performs sentence-level topic and sentiment analysis using transformer models.

-   **Script:** `scripts/run_emb_roberta.py`
-   **Process:**
    1.  **Topic Classification:** Assigns a theme to each article using `sentence-transformers` embeddings.
    2.  **Sentence Extraction:** Splits articles into sentences and filters them based on keywords related to public administration.
    3.  **Sentiment Analysis:** Predicts the sentiment of filtered sentences using a pre-trained XLM-RoBERTa sentiment model.
-   **Outputs:** Several files, including `data/processed/swissdox_sentences_with_sentiment.parquet`.

#### B) The LLM Pipeline (Apertus)

This pipeline annotates each article at the document level using a locally hosted LLM.

-   **Script:** `scripts/run_llm_apertus.py`
-   **Long-Running Jobs:** For large datasets, use the `scripts/run_llm_apertus_overnight.py` wrapper, which provides robust checkpointing and automatically restarts the process to manage resources and ensure completion.
-   **Process:** For each article, the LLM identifies the primary topic, sentiment towards public administration, and populist rhetoric, along with justifications.
-   **Output:** `data/processed/swissdox_articles_labeled_llm.parquet`

### Step 3: Compare Annotations

Align and compare the outputs from different pipelines (e.g., `Emb+RoBERTa` vs. `LLM Apertus` vs. an external annotation file).

-   **Script:** `scripts/run_comparison.py`
-   **Process:** Ingests analysis results, harmonizes labels, and computes agreement metrics and confusion matrices.
-   **Outputs:**
    -   An aligned dataset (`data/processed/comparison/aligned_three_models.parquet`).
    -   An agreement report with Accuracy and Cohen's Kappa (`data/processed/comparison/agreement_report.csv`).
    -   Confusion matrices for topic classification.

### Step 4: Generate Plots

Create a suite of publication-ready visualizations from the analysis results.

-   **Script:** `scripts/run_comparison_plots.py`
-   **Process:** Generates time-series and compositional plots for sentiment, topics, and populism.
-   **Example Usage:**
    ```bash
    python scripts/run_comparison_plots.py \
      --a data/processed/swissdox_articles_labeled_llm.parquet \
      --b data/external/GPT_ANALYSIS.csv \
      --c data/processed/swissdox_sentences_with_sentiment.parquet \
      --outdir outputs/plots \
      --no-show
    ```
-   **Generated Plots Include:**
    -   Sentiment shares over time (monthly).
    -   Stacked topic shares over time (monthly).
    -   Monthly negativity rate combined with department composition.
    -   Populism "YES" share over time.

## Design Principles

-   **Modularity:** Strict separation between reusable logic (`src/`) and executable scripts (`scripts/`).
-   **Determinism:** Pipelines are designed to be deterministic and resumable.
-   **Harmonization:** Label spaces are harmonized across different models for consistent comparison.
-   **Performance:** The Parquet file format is used internally for efficient I/O and data handling.

## Notes

-   Large data files (e.g., in `data/` and `outputs/`) are excluded from version control via `.gitignore`.
-   The LLM pipeline requires a locally running OpenAI-compatible server.
-   This repository is intended for academic research use.
