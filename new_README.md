# Information flows, but can you catch it?

> **Milestone P3: Final Project**. This repository contains the final execution of our NLP project, exploring the semantic links between podcasts and their correlation with financial markets.

## Project Report

For a detailed description of our methodology, findings, and analysis, please refer to our final report:
**[Read the Final Report (report.pdf)](report.pdf)**

## Abstract

We explored how podcasts are linked by shared "ideas" and whether these informational signals can anticipate financial market movements. Using the **SPoRC dataset** (Structured Podcast Research Corpus), we generated embeddings for podcast transcripts to place episodes and abstract "ideas" into a shared semantic space.

**Key Implementations in P3:**

- **Idea Extraction**: We embedded texts and episodes to calculate relevance scores, tracking specific "idea intensity" over time.
- **Graph Analysis**: We constructed knowledge graphs using `pyvis` to visualize the hidden connections between podcasts, hosts, and topics.
- **Market Correlation**: We analyzed the time-series of these idea scores against stock market data (via `yfinance`) to identify potential predictive signals.

This repository hosts the complete codebase for this pipeline.

## Team Contributions

_(Please update the table below with your specific group members and their contributions)_

| Team Member      | Contributions |
| :--------------- | :------------ |
| **Vlad Vladutu** | []            |
| **Arne Wiese**   | []            |

## Repository Structure

Per the P3 requirements, the main logic is consolidated into a single notebook, with helper functions modularized.

- `main.ipynb` - **The Core Logic**. Contains the full pipeline:
  1.  Dataset visualization
  2.  Dataset preprocessing
  3.  "Idea" extraction (embeddings)
  4.  Preview generation
  5.  Graph construction
  6.  Graph visualization
  7.  Stock Market Correlation
- `utils/` - Contains external scripts and helper functions (e.g., `load_dataset.py`) called by the main notebook.
- `report.pdf` - The final project report.
- `assets/` - Static assets (images/icons) for notebook previews.
- `features/` - Preprocessed versioned features for reproducibility.

## Usage / Quickstart

The core logic is contained entirely within `main.ipynb`.

1. **Create and activate the conda environment**:
   ```bash
   conda create -n nlp_project python=3.10 -y
   conda activate nlp_project
   ```
2. **Install Dependencies**:
   ```python
   pip install -r requirements.txt
   ```
3. **Create Hugging Face Access Token**:
   Create a valid Hugging Face token and insert into main.pynb.
4. **Run Pipeline**:
   Execute `main.ipynb` sequentially to reproduce the embeddings, knowledge graphs, and correlation plots.

## 🔗 External Resources

Any additional large artifacts (model weights, processed datasets) used in this project are hosted here:

- **[Google Drive Folder Link](https://drive.google.com/drive/folders/1QABHakQ12phfQRyPg9b7-yrZjMOL3j6z)**

---

_Group 23 - NLP Project_
