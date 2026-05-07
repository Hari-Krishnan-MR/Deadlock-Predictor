# Deadlock Predictor 🔴

> Can we predict distributed system failures *before* they happen?
> That's exactly what this project tries to answer.

I built this as part of my MSc Data Analytics dissertation at 
National College of Ireland, but honestly it grew into something 
I'm genuinely proud of. The idea is simple — instead of waiting 
for a task to fail in a distributed system and then reacting, 
what if we could see it coming?

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.ai)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.969-brightgreen)]()
[![F1](https://img.shields.io/badge/F1%20Score-0.830-brightgreen)]()
[![Dataset](https://img.shields.io/badge/Dataset-Google%20Cluster%20Trace%202011-lightgrey)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)]()

---

## The Problem

In large-scale distributed systems like Google's Borg, 
thousands of tasks run simultaneously across thousands 
of machines. Sometimes tasks fail — they get killed, 
evicted, or crash due to resource contention. By the 
time you know about it, the damage is done.

Traditional approaches only detect deadlocks **after** 
they occur. I wanted to predict them **before** — using 
signals already present in the system's runtime behaviour.

---

## What I Did

I took the Google Cluster Trace 2011 — 1.83 million 
task events from a real production cluster — and asked:
*"what patterns in a task's runtime behaviour predict 
whether it will fail?"*

The answer turned out to be six novel features I 
engineered myself from the raw event logs. None of 
these exist in the original dataset — I had to build 
them from scratch by joining events across time.

---

## Results

| Model | F1 Score | ROC-AUC | Accuracy |
|-------|----------|---------|----------|
| Logistic Regression (baseline) | 0.440 | 0.692 | 61.7% |
| Random Forest | 0.793 | 0.957 | 89.8% |
| **XGBoost — best model** | **0.830** | **0.969** | **91.6%** |

**Cross-validation:** F1 = 0.8319 ± 0.0024 across 5 folds  
**Statistical significance:** t = 244.14, p < 0.001  
**Operational impact:** catches 85 of every 100 failures proactively

---

## The Six Novel Features

This is the part I'm most proud of. Each feature required 
joining events across time — none exist in a single row 
of the raw data.

| Feature | What It Captures | F1 Drop if Removed |
|---------|-----------------|-------------------|
| `execution_duration_sec` | Time from schedule to final event | **-5.9%** |
| `scheduling_delay_sec` | How long scheduler took to assign machine | -2.3% |
| `resource_overcommit_ratio` | CPU vs machine average | -1.9% |
| `concurrency_density` | Tasks per machine per 300s window | -1.9% |
| `priority_inversion_flag` | Priority gap severity | -1.9% |
| `machine_failure_rate` | Historical machine reliability | -1.4% |

The most interesting finding was the bimodal pattern 
in `execution_duration_sec` — deadlock-like tasks 
either die very quickly (median 539s) or run 
for a very long time before being killed (mean 4,840s). 
Two completely different failure mechanisms hiding 
in one feature.

---

## Dataset

I used the Google Cluster Trace 2011, a publicly 
available 29-day production workload log from a 
Google Borg cluster.

- **1,827,856** raw task events (10 of 500 parts)
- **300,644** labelled tasks
- **12,509** machines tracked
- **76% normal, 24% deadlock-like**
- Licence: CC-BY (free to use)

```bash
# Download the data yourself
wget https://storage.googleapis.com/clusterdata-2011-2/task_events/part-00000-of-00500.csv.gz
```

See `data/README.md` for full download instructions.

---

## How to Run

### Option 1 — Kaggle (recommended, zero setup)

| Notebook | Link |
|----------|------|
| 01 Data Loading & Cleaning | [Open on Kaggle](https://www.kaggle.com/code/hharikrishnanmr/notebook-1a) |
| 02 Feature Engineering | [Open on Kaggle](https://www.kaggle.com/code/hharikrishnanmr/notebook-2) |
| 03 Modelling & Results | [Open on Kaggle](https://www.kaggle.com/code/hharikrishnanmr/notebook-3) |

Just click, hit Run All, and everything works. 
No setup, no login needed to view outputs.

### Option 2 — Run Locally

```bash
# Clone the repo
git clone https://github.com/Hari-Krishnan-MR/deadlock-predictor.git
cd deadlock-predictor

# Install dependencies
pip install -r requirements.txt

# Download data (see data/README.md)

# Run notebooks in order
jupyter notebook notebooks/01_data_loading_cleaning.ipynb
jupyter notebook notebooks/02_feature_engineering.ipynb
jupyter notebook notebooks/03_modelling_results.ipynb
```

### Option 3 — Use as a Python package

```python
from src.preprocessing import load_trace_parts, clean_trace
from src.features import build_feature_matrix
from src.models import train_xgboost, predict
from src.evaluate import evaluate_model

# Load and clean data
df_raw   = load_trace_parts('data/', n_parts=10)
df_clean = clean_trace(df_raw)

# Train best model
model = train_xgboost(X_train, y_train)

# Evaluate
results = evaluate_model(
    'XGBoost', y_test, y_pred, y_prob
)
```

---

## Project Structure    

deadlock-predictor/
│
├── notebooks/
│   ├── 01_data_loading_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modelling_results.ipynb
│
├── src/
│   ├── preprocessing.py   ← data loading and cleaning
│   ├── features.py        ← 6 novel feature functions
│   ├── models.py          ← LR, RF, XGBoost training
│   └── evaluate.py        ← metrics, ablation, CV
│
├── data/
│   ├── README.md          ← download instructions
│   └── sample/            ← 100 row sample for testing
│
├── outputs/
│   ├── figures/           ← all plots
│   └── results/           ← CSV result files
│
├── paper/
│   └── deadlock_prediction_paper.pdf
│
├── presentation/
│   └── DM_ML_Presentation.pptx
│
├── requirements.txt
└── .gitignore

---

## Key Findings

**1. Execution duration is the dominant predictor**  
Removing `execution_duration_sec` alone drops F1 by 
5.9% — more than any other single feature. It captures 
a bimodal failure pattern that no raw feature can see.

**2. Priority matters more than resource demand**  
Deadlock-like tasks actually request *more* memory 
on average but have *lower* priority. The scheduler 
kills low-priority tasks to reclaim resources, 
regardless of how much they ask for.

**3. All six novel features contribute positively**  
The ablation study confirms every single engineered 
feature improves performance independently — 
no redundancy, no noise.

**4. Results are statistically solid**  
F1 = 0.8319 ± 0.0024 across 5 folds with 
t = 244.14, p < 0.001. Not a fluke.

---

## What's Next — Deadlock-2.0

This is version 1.0. Here is what I plan to add:

- [ ] Full 500-part trace (225M rows)
- [ ] LSTM sequence model for temporal patterns
- [ ] Real-time scoring API (FastAPI)
- [ ] SHAP explainability dashboard
- [ ] Cross-dataset validation (Alibaba Cluster Trace)
- [ ] Conference paper submission

---

## Tech Stack

- Python 3.12
- pandas, numpy, scipy
- scikit-learn, XGBoost
- matplotlib, seaborn
- Kaggle Notebooks
- Google Cluster Trace 2011

---

## About

**Hari Krishnan M R**  
MSc Data Analytics — National College of Ireland  
[Kaggle](https://www.kaggle.com/hharikrishnanmr) | 
[GitHub](https://github.com/Hari-Krishnan-MR)

---

## Version History

| Version | Date | What Changed |
|---------|------|-------------|
| 1.0.0 | May 2026 | Initial release — 3 models, 6 novel features, ablation study |

