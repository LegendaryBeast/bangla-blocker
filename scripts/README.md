# Evaluation & Analysis Scripts

This directory contains standalone Python routines for running lexical baselines, generating figures, analyzing ablations, and verifying reproducibility.

---

## 🛠️ Script Overview

| Script | Purpose | Usage |
| :--- | :--- | :--- |
| [`run_tfidf_baselines.py`](run_tfidf_baselines.py) | Train and evaluate TF-IDF + LR and TF-IDF + SVM baselines across all 4 linguistic forms. | `python3 scripts/run_tfidf_baselines.py --data_dir bangla-blocker-dataset/splits_v2` |
| [`ablation_analysis.py`](ablation_analysis.py) | Statistical analysis of attention mechanisms and ordinal risk heads. | `python3 scripts/ablation_analysis.py` |
| [`generate_paper_figures.py`](generate_paper_figures.py) | Generates publication-resolution figures for the manuscript. | `python3 scripts/generate_paper_figures.py` |
| [`plot_lodo_barchart.py`](plot_lodo_barchart.py) | Plots Leave-One-Domain-Out (LODO) transfer degradation bars. | `python3 scripts/plot_lodo_barchart.py` |
| [`plot_three_class_comparison.py`](plot_three_class_comparison.py) | Visualizes 3-class decision routing (ALLOW / REVIEW / BLOCK) across models. | `python3 scripts/plot_three_class_comparison.py` |
| [`generate_domain_asymmetry_figure.py`](generate_domain_asymmetry_figure.py) | Analyzes intent distribution across Banking, Healthcare, and E-commerce. | `python3 scripts/generate_domain_asymmetry_figure.py` |
| [`apply_dataset_audit_fixes.py`](apply_dataset_audit_fixes.py) | Audit verification and leakage-control filtering logic. | `python3 scripts/apply_dataset_audit_fixes.py` |

---

## 💻 Environment Setup

Run with dependencies installed from `requirements.txt`:
```bash
pip install -r requirements.txt
```
