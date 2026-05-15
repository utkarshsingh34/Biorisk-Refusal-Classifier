# Biology Refusal Classifier

Binary text classifier flagging natural-language prompts for refusal on biosecurity grounds. Test task submission.

**Write-up:** [`biorisk-refusal-classifier-writeup.docx`](./biorisk-refusal-classifier-writeup.docx) (1–2 pages, the short note covering data choices, modeling, results, future work).

---

## Quickstart

The pipeline runs end-to-end on Google Colab free tier. To reproduce:

1. **Open** `notebooks/01_data_pipeline.ipynb` in Colab ([Open in Colab](https://colab.research.google.com/github/utkarshsingh34/Biorisk-Refusal-Classifier/blob/main/notebooks/01_data_pipeline.ipynb)).
3. **Mount Google Drive** (first cell of every notebook). Data and model outputs persist at `/content/drive/MyDrive/biology_refusal/`.
4. **Run cells top to bottom.** No code edits needed.
5. **For DistilBERT training** (notebook 02 and 06): switch Colab runtime to **T4 GPU** (`Runtime → Change runtime type → T4 GPU`).

Total runtime end-to-end: ~30 minutes (data ~5 min, modeling ~15 min with GPU, leakage check ~3 min, cleanup re-eval ~10 min, evaluation viz ~2 min, temperature scaling ~10 min).

---

## Notebooks

Run in order; each persists outputs to Drive that downstream notebooks read.

| # | Notebook | What it does | Runtime |
|---|---|---|---|
| 01 | `01_data_pipeline.ipynb` | Loads WMDP-bio + 4 negative buckets (MMLU bio/non-bio, Alpaca, dual_use). KMeans topic-clustered split on WMDP. Per-bucket stratified split on negatives. | ~5 min CPU |
| 02 | `02_modeling.ipynb` | Trains TF-IDF+LR, MiniLM+LR, DistilBERT on the same split. Shared evaluation function. Per-bucket recall, PR curves, operating points. | ~15 min GPU |
| 03 | `03_leakage_check.ipynb` | Six diagnostic checks: exact dups, near-dups via embeddings, cross-source proximity, label sanity, top discriminating features, bucket-pair separability. | ~3 min CPU |
| 04 | `04_cleanup_and_reeval.ipynb` | Drops near-duplicate test examples (sim ≥ 0.90). Augments TF-IDF stopwords with format/imperative artifacts. Retrains all three models on cleaned data. | ~10 min GPU |
| 05 | `05_evaluation.ipynb` | Produces all figures and consolidated metrics tables from the cleaned predictions. | ~2 min CPU |
| 06 | `06_temperature_scaling.ipynb` | Fits a temperature scalar on a held-out 15% of training data, retrains DistilBERT on the remaining 85%, applies scaling. Demonstrated the standard calibration fix wasn't needed (model already well-calibrated). | ~10 min GPU |

---

## Data sources

All from public HuggingFace datasets. No scraping, no auth required.

| Source | Role | Loaded from |
|---|---|---|
| `cais/wmdp` (wmdp-bio) | Positives (refuse) | HuggingFace |
| `cais/mmlu` (college_biology, high_school_biology) | Hard negatives (mmlu_bio) | HuggingFace |
| `cais/mmlu` (chemistry, physics, medicine, clinical) | Format-control negatives (mmlu_other) | HuggingFace |
| `tatsu-lab/alpaca` | Baseline negatives (alpaca) | HuggingFace |
| `cais/mmlu` (virology, medical_genetics, anatomy) + `qiaojin/PubMedQA` (filtered) | Adjacency negatives (dual_use) | HuggingFace |

WMDP-bio answer choices are stripped at load time per task instructions.

---

## Metrics summary

Held-out test set (post-cleanup, n=1,294):

| Model | Acc | F1 (refuse) | F1 (don't refuse) | AP | ROC-AUC | R@FPR=1% |
|---|---|---|---|---|---|---|
| TF-IDF + LR | 0.971 | 0.938 | 0.981 | 0.980 | 0.993 | 0.825 |
| MiniLM + LR | 0.971 | 0.941 | 0.981 | 0.979 | 0.994 | 0.789 |
| DistilBERT | **0.985** | **0.968** | **0.990** | **0.994** | **0.997** | **0.967** |

Per-bucket recall on negatives (the headline diagnostic — higher = better content discrimination on hard slices):

| Model | alpaca | mmlu_other | mmlu_bio | dual_use |
|---|---|---|---|---|
| TF-IDF | 0.997 | 0.985 | 0.931 | 0.899 |
| MiniLM | 0.998 | 0.985 | 0.961 | **0.808** |
| DistilBERT | **1.000** | **1.000** | **0.990** | **0.960** |

Full per-class precision / recall / F1 / support tables are produced in notebook 04. The consolidated table is saved to `results/results_table.csv` by notebook 05.

---

## Key findings

1. **The negative-bucket design did real work.** Difficulty gradient is real and monotonic — alpaca (1.00) → mmlu_other (0.99) → mmlu_bio (0.93–0.99) → dual_use (0.81–0.96).
2. **DistilBERT wins on production-relevant metrics.** At FPR ≤ 1%, recall holds at 0.967 vs 0.79–0.83 for linear models. Aggregate F1 understates the gap.
3. **MiniLM + LR underperforms TF-IDF on dual_use (0.81 vs 0.90).** Frozen semantic embeddings + linear classifier worse than lexical features on the hardest slice — non-obvious result about when "smart features + simple model" loses to "simple features + simple model."
4. **Models were learning content, not format.** Leakage diagnostic surfaced format-artifact features in TF-IDF; remediation (stopwords + near-dup test cleanup) moved metrics by <0.01. Validates that headline numbers reflect content discrimination.
5. **DistilBERT was already well-calibrated.** Temperature scaling fit T=1.20, ECE moved from 0.0045 to 0.0067 — the visual bimodality reflects honest confidence, not miscalibration.

---

## Repo layout

```
biology-refusal-classifier/
├── README.md                     ← you are here
├── biorisk-refusal-classifier-writeup.docx                  ← ~1 page note
├── notebooks/                    ← 6 Colab-ready notebooks (run in order)
├── figures/                      ← evaluation figures (PNGs)
└── results/
    ├── results_table.csv         ← consolidated metrics
    └── all_models_wrong.csv      ← examples missed by all three models
```

---
