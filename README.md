# Information flows, but can you catch it?
**Milestone P3: Final Project** — NLP pipeline for mapping semantic links between podcast discourse and (exploratory) alignment with financial markets.

## Project Report
For full methodology, results, and figures, read:
**[Final Report (report.pdf)](report.pdf)**

## Brief Project Explanation
> Read the full Report for a better understanding

This project uses the **SPoRC dataset** (May–June 2020) to study how *themes (“ideas”)* appear across podcast episodes and how those themes connect shows, hosts, and categories in a semantic space.

The pipeline does three things:

1. **Idea extraction (theme intensity over time)**  
   - Transcripts are chunked (fixed-length windows with overlap) to avoid context-length limits.  
   - A sentence embedding model places both **idea descriptions** and **transcript chunks** into the same vector space.  
   - “Idea intensity” is computed by aggregating chunk-level cosine similarities into an episode score, then averaging into a daily time series.

2. **Graph construction (semantic neighborhoods)**  
   - Sparse similarity graphs are built at **episode**, **podcast**, and **host** level.  
   - Edges exist only above a similarity threshold to keep graphs interpretable.  
   - Result: coherent clusters plus cross-category bridges that help navigate large collections.

3. **Market alignment (exploratory, not predictive claims)**  
   - Daily idea intensity is aligned with market variables (returns and volume).  
   - We compute Pearson/Spearman correlations (with p-values) and run Granger-causality tests across lags.  
   - Main takeaway: **robust lead–lag evidence is weak** in this short time window; signals are often **reactive rather than predictive**, and aggressive pooling can create misleading patterns.

If you write “we predict markets,” you did not read the report.

## Key Implementations in P3
- **Preprocessing**: filter metadata fields, coarse category filtering, exclude low-coverage days, fixed per-day sampling for stable time series.
- **Embeddings**: chunk-level scoring + multiple aggregation strategies (mean/max/top-k/log-mean-exp/pct-above-threshold) to control “spiky” vs “broad” evidence.
- **Idea generation (optional qualitative aid)**: evidence-grounded, headline-style summaries from selected sentences (MMR + Flan-T5).
- **Graphs**: thresholded semantic similarity networks with shortest-path support via inverse-similarity edge lengths.
- **Market tests**: aligned time series + correlation scans + Granger tests across lags; interpretation kept explicitly exploratory.

## Repository Structure
Per P3 requirements, the main logic is consolidated into a single notebook.

- `main.ipynb` - **Core pipeline**:
  1. Dataset visualization
  2. Dataset preprocessing
  3. Idea extraction (chunk embeddings + scoring)
  4. (Optional) Idea/preview generation
  5. Graph construction
  6. Graph visualization
  7. Market alignment (correlation + Granger tests)
- `report.pdf` - Final report (methods + results).
- `report_latex.zip` - The source files for the report
- `assets/` - Images/icons used for notebook previews.
- `features/` - Preprocessed/versioned features for reproducibility. The versioning README.md is also displayed in the code.
- `drafts/` - A lot of drafts

## Usage / Quickstart
The pipeline is entirely in `main.ipynb`.

1. **Create and activate the conda environment**
   ```bash
   conda create -n nlp_project python=3.13.1 -y
   conda activate nlp_project
   ```

2. **Install dependencies**

   * Installng the packages required by the notebook happens in the first cell.
   * **OR** use requirements.txt

3. **Set Hugging Face access token**

   * Create a valid Hugging Face token and insert it where the notebook expects it.

4. **Download large artifacts**

   * Download embeddings (`v0.1.6`) from the external resources in the relative `features/episode_vecs_v0.1.6` directory.
   * The dataset is huge. Make sure you download it at most once, and specify the correct path.

5. **Run**

   * Execute `main.ipynb` top-to-bottom (sequential execution).
     Skipping cells could break reproducibility. Be cautious!

## Team Contributions

| Team Member      | Contributions                                                                                                                                                 |
| :--------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Vlad Vladutu** | most of the codebase, technical writing and correctness pass on the report (methods, metrics, numbers validation), full LaTeX refinement and final formatting |
| **Arne Wiese**   | initial graph visualization, conducted experiments on different embeddings, created extra report figures/visuals, initial LaTeX report + README               |

## 🔗 External Resources

Large artifacts (The episode embeddings which are to big to host on github, i.e `v0.1.6`) are hosted here:

* **Google Drive Folder**: [https://drive.google.com/drive/folders/1TejVi77wSzXLJGDSFDHR_nPDnm6s3Ovb](https://drive.google.com/drive/folders/1TejVi77wSzXLJGDSFDHR_nPDnm6s3Ovb)
