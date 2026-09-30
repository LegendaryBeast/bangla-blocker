# BanglaBlocker: Experimental Notebooks Guide

This directory contains the complete, reproducible suite of Jupyter notebooks for evaluating baselines, training BanglaBlocker models, running cross-domain transfer audits, and conducting ablation experiments.

---

## 📋 Experiment & Paper Mapping

| Notebook | Focus & Model | Paper Mapping | Execution Platform |
| :--- | :--- | :--- | :--- |
| [`01_baseline_banglabert.ipynb`](01_baseline_banglabert.ipynb) | **BanglaBERT (110M)** fine-tuning & multi-form evaluation | **Table 12, Table 13** | Kaggle / Colab (GPU) |
| [`02_baseline_singguard.ipynb`](02_baseline_singguard.ipynb) | **SingGuard** zero-shot safety guardrail evaluation | **Table 12, Table 15** | Kaggle (T4/P100) |
| [`03_baseline_qwen3guard.ipynb`](03_baseline_qwen3guard.ipynb) | **Qwen3Guard-Gen-0.6B** zero-shot evaluation | **Table 12, Table 15** | Kaggle (T4/P100) |
| [`04_baseline_indicguard.ipynb`](04_baseline_indicguard.ipynb) | **IndicGuard** multilingual guardrail baseline | **Table 12, Table 15** | Kaggle (GPU) |
| [`05_baseline_tigerllm_lora.ipynb`](05_baseline_tigerllm_lora.ipynb) | **TigerLLM-1B (LoRA)** generative decoder baseline | **Table 12, Table 14, Table 16** | Kaggle (A100 / 2×T4) |
| [`06_banglablocker_base_training.ipynb`](06_banglablocker_base_training.ipynb) | **BanglaBlocker-Base (281M)** adapted training & eval | **Table 12, Table 15, Table 17** | Kaggle (1× T4) |
| [`07_banglablocker_large_training.ipynb`](07_banglablocker_large_training.ipynb) | **BanglaBlocker-Large (564M)** adapted training & eval | **Table 12, Table 15, Table 17** | Kaggle (1× T4 / P100) |
| [`08_cross_domain_lodo_transfer.ipynb`](08_cross_domain_lodo_transfer.ipynb) | **Leave-One-Domain-Out (LODO)** generalization | **Table 16, Fig. 6** | Kaggle / Colab |
| [`09_ablation_study.ipynb`](09_ablation_study.ipynb) | **Ablation Study**: Token attention & ordinal loss | **Table 10** | Kaggle (T4) |
| [`10_banglablocker_full_benchmark_colab.ipynb`](10_banglablocker_full_benchmark_colab.ipynb) | **End-to-End Pipeline**: Full benchmark & LODO | **All Tables** | Google Colab |

---

## 🚀 Execution Instructions

### Dataset Dependency
All notebooks expect the certified, leakage-controlled partition files from `bangla-blocker-dataset/splits_v2/`:
- `train.csv` (2,850 scenarios / 11,400 prompts)
- `val.csv` (450 scenarios / 1,800 prompts)
- `heldout_test.csv` (900 scenarios / 3,600 prompts)
- `decision_matched_test.csv` (270 scenarios / 1,080 prompts)

When running on **Kaggle** or **Google Colab**:
1. Upload `bangla-blocker-dataset/splits_v2.zip` as a dataset.
2. The notebooks automatically locate the path from Kaggle's `/kaggle/input/` or Colab's `/content/` root.

### Random Seed Protocol
To reproduce reported standard deviations ($\pm \sigma$), neural models are evaluated across seeds **42, 43, 44**.

---

## 🗄️ Legacy & Development Archive
The [`archive/`](archive/) folder contains original, unsorted development iterations, smoke tests, and hardware-specific scratch versions retained for full historical auditability.
