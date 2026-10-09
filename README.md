# Adaptive Fraud Detection System: Dual-Engine Architecture & Drift Management

This repository contains the complete implementation of a production-ready, self-monitoring fraud detection system built on the IEEE-CIS Fraud Detection dataset. It features a dual-engine machine learning architecture, financial cost-based decision logic, and an automated Population Stability Index (PSI) monitor that triggers incremental online retraining when concept drift occurs.

---

## 1. Setup and Dependency Installation

### Prerequisites

* Python 3.9+
* Kaggle API (for downloading the dataset)
* Minimum 16GB RAM (32GB recommended for the unoptimized Kaggle dataset)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/adaptive-fraud-detection.git
cd adaptive-fraud-detection

```


2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```


3. Install dependencies:
```bash
pip install pandas numpy scikit-learn xgboost tensorflow matplotlib kaggle

```



### Data Acquisition

The system uses the IEEE-CIS Fraud Detection dataset. Download it directly using the Kaggle API:

```bash
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip -d data/

```

---

## 2. Approach, Key Decisions, and Results (Write-up)

### The Problem

Static fraud classifiers degrade silently in production. Fraudsters constantly change tactics (concept drift) and macroeconomic factors shift (data drift). Furthermore, naive models optimize for statistical metrics (like ROC-AUC) on highly imbalanced data, missing the true financial impact of False Positives (friction) vs. False Negatives (financial loss).

### The Approach

The architecture is designed around four core pillars:

1. **Dual-Engine Prediction:** A Supervised XGBoost model learns historical fraud patterns. An Unsupervised Deep Autoencoder, trained exclusively on legitimate transactions, acts as a safety net to catch zero-day anomalies by calculating reconstruction error (MSE).
2. **Meta-Learner Stacking:** A Logistic Regression meta-learner bridges the probabilistic output of Model A and the unbounded MSE of Model B, learning the optimal combination of both.
3. **Cost-Optimized Policy Layer:** Instead of defaulting to a $0.5$ probability threshold, the system dynamically calculates the decision boundary that minimizes expected financial loss.
4. **Drift Detection & Response:** A continuous monitor calculates the Population Stability Index (PSI) on high-feature-importance covariates. If PSI exceeds 0.2, the system simulates an incremental update (`xgb.fit(xgb_model=booster)`) to patch the model without a full historical retrain.

### Key Architectural Decisions

* **Temporal Splitting over Random CV:** Fraud data must be split chronologically (`shuffle=False`). Random splitting leaks "future" fraud tactics into the training set, causing catastrophic overfitting.
* **Algorithmic Imbalance Handling over SMOTE:** Synthetic oversampling interpolates between distinct fraud vectors, creating unrealistic noise. We opted for algorithm-level `scale_pos_weight` in XGBoost and strict anomaly-framing for the Neural Network.
* **PR-AUC over ROC-AUC:** Because legitimate transactions outnumber fraud 97-to-3, the False Positive Rate denominator is massive, artificially inflating ROC-AUC. We strictly use Precision-Recall AUC (PR-AUC) to penalize false alarms.
* **PSI over KS-Test:** The Kolmogorov-Smirnov test is overly sensitive in massive datasets, flagging benign statistical variances. PSI provides a stable, bucketed metric favored in banking.

### Results

* **Baseline Supervised (XGBoost):** ROC-AUC: 0.9022 | PR-AUC: 0.5061
* **Baseline Unsupervised (Autoencoder):** PR-AUC: 0.0788 (Expected for unsupervised isolation)
* **Financial Optimization:** Applying the cost matrix ($50/FP, $200/FN) via the meta-learner shifted the optimal decision threshold to `0.91`, reducing expected operational loss on the validation set from **$972,500** (default 0.5 threshold) to **$521,900** — a simulated savings of $450,600.
* **Drift Recovery:** Injected a 100-row zero-day attack slice, crashing the stale model's PR-AUC to `0.0189`. After automated incremental retraining, performance on the attack slice recovered to `1.0000`.

---

## 3. How to Reproduce Results

Run the main pipeline script, which executes the entire workflow sequentially.

```bash
python run_pipeline.py

```

*(Alternatively, execute the cells chronologically if using the provided Jupyter Notebook).*

The pipeline executes the following stages:

1. **Data Ingestion & Memory Optimization:** Merges `transaction` and `identity` tables, then executes a safe downcasting function (float64 $\rightarrow$ float32/int8) to prevent Out-Of-Memory errors. Categoricals are Label Encoded; missing numeric values are imputed with `-999`.
2. **Supervised Training:** Trains the XGBoost classifier on the first 80% of the timeline.
3. **Unsupervised Training:** Scales the data (`MinMaxScaler`), isolates legitimate transactions, and trains the Deep Autoencoder.
4. **Meta-Ensembling:** Scales the Autoencoder MSE and trains the Logistic Regression meta-learner on the validation set outputs.
5. **Cost Optimization:** Evaluates the ensemble probabilities against the custom cost function to output the optimal decision threshold.

---

## 4. Instructions for Re-Running Evaluation and Drift Simulations

### Evaluating New Data

To evaluate a new batch of data without retraining the models, use the standalone evaluation script:

```bash
python evaluate_batch.py --input data/new_transactions.csv --weights models/

```

This script will output the PR-AUC, confusion matrix, and expected financial cost based on the previously saved optimal threshold.

### Modifying the Financial Cost Matrix

If your business logic changes (e.g., support call costs rise to $75), you can re-run the threshold optimizer without retraining the heavy base models:

```python
# In optimize_threshold.py
optimal_thresh, min_cost = optimize_threshold(
    y_true, 
    final_ensemble_probs, 
    cost_fp=75,  # Update False Positive cost
    cost_fn=200  # Update False Negative cost
)

```

### Simulating Drift and Online Retraining

To re-run the concept drift simulation and test the system's patching capability:

1. Open `drift_simulation.py`.
2. Modify the `attack_indices` parameters (e.g., change which features spike or drop to simulate different attack vectors).
3. Execute the script to view the PSI monitor alert, the performance drop, and the incremental tree-boosting recovery.

```bash
python drift_simulation.py

```
