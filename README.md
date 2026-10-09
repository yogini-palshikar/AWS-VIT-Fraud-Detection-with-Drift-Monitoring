# Fraud Detection System: Dual-Engine Architecture & Drift Management

This repository contains an end-to-end, production-style machine learning pipeline for fraud detection, built on the IEEE-CIS Fraud Detection dataset. It features a dual-engine architecture, financial cost-based decision logic, and an automated Population Stability Index (PSI) monitor that triggers incremental online retraining when concept drift occurs.

The entire workflow—from memory-optimized data ingestion to drift simulation—is contained within a single, comprehensive Jupyter Notebook for easy exploration and reproducibility.

---

## 1. Setup and Dependency Installation

### Prerequisites

* Python 3.9+
* Jupyter Notebook or JupyterLab (or run directly via Google Colab / Kaggle)
* Minimum 16GB RAM (32GB recommended due to the unoptimized Kaggle dataset size)

### Installation

1. Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/yourusername/adaptive-fraud-detection.git
cd adaptive-fraud-detection

```


2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```


3. Install the required dependencies:
```bash
pip install pandas numpy scikit-learn xgboost tensorflow matplotlib jupyter

```



### Data Acquisition

The system uses the IEEE-CIS Fraud Detection dataset. Download it using the Kaggle API and extract it into a `data/` folder in your project directory:

```bash
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip -d data/

```

---

## 2. Approach, Key Decisions, and Results

### The Problem

Static fraud classifiers degrade silently in production. Fraudsters constantly change tactics (concept drift) and macroeconomic factors shift (data drift). Furthermore, naive models optimize for statistical metrics (like ROC-AUC) on highly imbalanced datasets, ignoring the true financial impact of False Positives (customer friction) versus False Negatives (financial loss).

### The Architecture

The pipeline is designed around four core pillars:

1. **Dual-Engine Prediction:** A Supervised XGBoost model learns historical fraud patterns. An Unsupervised Deep Autoencoder, trained exclusively on legitimate transactions, acts as a safety net to catch zero-day anomalies by calculating reconstruction error (MSE).
2. **Meta-Learner Stacking:** A Logistic Regression meta-learner bridges the probabilistic output of Model A and the unbounded MSE of Model B, learning the optimal mathematical combination of both.
3. **Cost-Optimized Policy Layer:** Instead of defaulting to a $0.5$ probability threshold, the system dynamically calculates the decision boundary that minimizes expected financial loss based on a custom business cost matrix.
4. **Drift Detection & Response:** A continuous monitor calculates the Population Stability Index (PSI) on high-importance covariates. If PSI exceeds 0.2, the system simulates an incremental update (`xgb.fit(xgb_model=booster)`) to patch the model without a full historical retrain.

### Key Technical Decisions

* **Temporal Splitting over Random CV:** Fraud data must be split chronologically (`shuffle=False`). Random splitting leaks "future" fraud tactics into the training set, causing catastrophic overfitting.
* **Algorithmic Imbalance Handling over SMOTE:** Synthetic oversampling interpolates between distinct fraud vectors, creating unrealistic noise. I utilized algorithm-level `scale_pos_weight` in XGBoost and strict anomaly-framing for the neural network.
* **PR-AUC over ROC-AUC:** Because legitimate transactions outnumber fraud 97-to-3, the False Positive Rate denominator is massive, artificially inflating ROC-AUC. I optimized strictly for Precision-Recall AUC (PR-AUC) to penalize false alarms.
* **PSI over KS-Test:** The Kolmogorov-Smirnov test is overly sensitive in massive datasets, flagging benign statistical variances. PSI provides a stable, bucketed metric favored in the financial industry.

### Experimental Results

* **Baseline Supervised (XGBoost):** ROC-AUC: 0.9022 | PR-AUC: 0.5061
* **Baseline Unsupervised (Autoencoder):** PR-AUC: 0.0788 (Expected for unsupervised isolation)
* **Financial Optimization:** Applying the cost matrix ($50/FP, $200/FN) via the meta-learner shifted the optimal decision threshold to `0.91`, reducing expected operational loss on the validation set from **$972,500** (default 0.5 threshold) to **$521,900** — a simulated savings of $450,600.
* **Drift Recovery:** Injected a 100-row zero-day attack slice, crashing the stale model's PR-AUC to `0.0189`. After automated incremental retraining, performance on the attack slice recovered to `1.0000`.

---

## 3. How to Reproduce Results

All code is structured sequentially inside the main Jupyter Notebook.

1. Launch Jupyter Notebook from your terminal:
```bash
jupyter notebook

```


2. Open the `.ipynb` file in this repository.
3. **Run the cells sequentially from top to bottom.**

The notebook is organized into the following distinct execution phases:

* **Phase 1: Data Ingestion & Memory Optimization:** Merges `transaction` and `identity` tables, then executes a safe downcasting function (float64 $\rightarrow$ float32/int8) to prevent Out-Of-Memory errors. Categoricals are Label Encoded; missing numeric values are imputed natively.
* **Phase 2: Supervised Training:** Trains the XGBoost classifier on the historical training window.
* **Phase 3: Unsupervised Training:** Scales the data, isolates legitimate transactions, and trains the Deep Autoencoder.
* **Phase 4: Meta-Ensembling & Cost Optimization:** Stacks the models, evaluates the ensemble against the financial cost function, and outputs the optimal decision threshold.
* **Phase 5: Drift Simulation & Online Retraining:** Calculates the PSI on a live data stream, triggers an alert upon detecting covariate shift, and executes the `xgb_model.get_booster()` incremental update to patch the model dynamically.

---

