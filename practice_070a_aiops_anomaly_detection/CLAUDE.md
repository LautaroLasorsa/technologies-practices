# Practice 070a -- AIOps: Anomaly Detection on Distributed Metrics

## Technologies

- **PyOD** (v2.0+) -- Python Outlier Detection library with 50+ algorithms under a unified `fit`/`predict` API. Includes both classical (LOF, KNN) and deep learning (AutoEncoder, VAE) detectors.
- **scikit-learn** -- Machine learning library; used here for `IsolationForest`, `StandardScaler`, and evaluation metrics (`roc_auc_score`, `precision_recall_curve`, `f1_score`).
- **pandas** -- DataFrame-based data manipulation for loading, transforming, and aggregating metrics data.
- **NumPy** -- Numerical computing; used for array operations, random generation of synthetic metrics, and statistical computations.
- **matplotlib** -- Plotting library for generating ROC curves, PR curves, and anomaly score distributions.

## Stack

Python 3.12+ (uv). No Docker needed.

## Theoretical Context

### What AIOps Is & The Problem It Solves

AIOps -- Artificial Intelligence for IT Operations -- was coined by [Gartner in 2016](https://www.gartner.com/en/information-technology/glossary/aiops-artificial-intelligence-operations) to describe the application of machine learning, big data analytics, and automation to IT operations processes. The core idea: modern distributed systems generate far more telemetry data (logs, metrics, traces) than any human operator can manually monitor. AIOps platforms use ML to automatically detect anomalies, correlate events across services, identify root causes, and trigger remediation -- shifting from reactive firefighting to proactive incident prevention.

The problem AIOps addresses is fundamentally one of **scale**. A typical microservice architecture with 50 services, each emitting 10 metrics at 1-second granularity, produces 500 data points per second -- 43 million per day. Manual threshold-based alerting (e.g., "alert if CPU > 80%") generates excessive false positives and misses subtle multi-dimensional anomalies like a gradual memory leak that only manifests as latency degradation under specific traffic patterns.

The global AIOps market is projected to grow from $27.6 billion (2024) to over $120 billion by 2033. Gartner predicted that by 2024, 40% of companies would use AIOps for monitoring their applications and infrastructure. This is not a niche -- it is becoming the standard approach to observability.

### Why Anomaly Detection Matters for Distributed Systems

In a distributed system, failures rarely manifest as a single component crashing. Instead, they appear as **subtle metric deviations** that cascade across service boundaries:

- **Latency spikes**: A database connection pool exhaustion in the payment service causes upstream API gateway timeouts, which triggers client retries, amplifying the load.
- **Error rate surges**: A bad deployment to one replica of the user-service causes 5% of requests to return 500s, but only for requests routed to that specific pod.
- **Resource exhaustion**: A memory leak in the order-service grows at 2MB/hour, invisible for days until the container hits its memory limit and gets OOM-killed during peak traffic.
- **Cascading failures**: A network partition between two availability zones causes the inventory-service to retry indefinitely, saturating the message queue, which starves the payment-service of events.

Traditional static thresholds fail for several reasons: metrics have diurnal patterns (traffic peaks at noon, dips at 3AM), seasonal trends (Black Friday), and service-specific baselines (a payment service's 200ms P99 is normal; the same latency for a cache lookup is catastrophic). Anomaly detection algorithms learn what "normal" looks like for each metric and service, and flag deviations from that learned baseline.

### Types of Anomalies

Anomaly detection literature classifies anomalies into three categories, each relevant to distributed systems:

**Point anomalies**: A single data point that deviates significantly from the rest. Example: a latency measurement of 15,000ms when the typical range is 50-200ms. This is the simplest case -- a Z-score or threshold check can catch it. Most alerting systems focus here.

**Contextual (conditional) anomalies**: A data point that is anomalous only in a specific context, but normal otherwise. Example: CPU usage of 95% is normal during a batch ETL job at 2AM, but anomalous at 2PM during steady-state traffic. Detecting these requires understanding the temporal or conditional context in which the metric exists.

**Collective anomalies**: A collection of data points that are individually normal but collectively indicate an anomaly. Example: each metric of the order-service is within normal bounds, but the combination of slightly elevated latency + slightly increased error rate + slightly higher memory usage together indicate an impending failure. Multivariate detectors (Isolation Forest, AutoEncoder) are needed to catch these.

### Statistical Baselines: When They Work and When They Fail

Before reaching for ML, it is worth understanding the statistical methods that form the foundation of most alerting systems:

**Z-score (standard score)**: Measures how many standard deviations a point is from the mean. For time series, a rolling Z-score computes mean and std over a sliding window. `z = (x - mean) / std`. Points where `|z| > threshold` (typically 3.0) are flagged. Works well for stationary data with roughly Gaussian distributions. Fails when data has trends, seasonality, or heavy tails.

**Moving average + band**: Compute a rolling mean and rolling standard deviation over a window. The "normal" band is `[mean - k*std, mean + k*std]`. Points outside the band are anomalies. This is essentially what Prometheus alerting and Datadog's basic anomaly monitors use. More robust to non-stationarity than raw Z-score because the window adapts to recent data.

**Exponential smoothing (EWMA)**: Weighted moving average that gives more weight to recent observations. `smoothed_t = alpha * x_t + (1 - alpha) * smoothed_{t-1}`. Used in control charts (EWMA charts). The advantage over simple moving average is faster response to level shifts.

**When statistical baselines fail**: (1) Multivariate anomalies where each dimension is individually normal but the combination is anomalous. (2) Complex nonlinear relationships between metrics that a linear model cannot capture. (3) High-dimensional data where the number of metric dimensions exceeds what manual threshold tuning can handle. This is where ML-based approaches add value.

### ML-Based Approaches

**Isolation Forest** ([Liu, Ting & Zhou, 2008](https://en.wikipedia.org/wiki/Isolation_forest)): The key insight is that anomalies are "few and different," making them easier to isolate via random partitioning. The algorithm builds an ensemble of isolation trees, where each tree recursively partitions data by randomly selecting a feature and a random split value within that feature's range. Anomalies, being rare and different from the majority, require fewer random splits to isolate -- resulting in shorter path lengths from root to leaf. The anomaly score is the average path length across all trees, normalized against the expected path length for the dataset size. Shorter average path = more anomalous. Key parameters: `n_estimators` (number of trees, default 100), `max_samples` (subsample size per tree), `contamination` (expected proportion of anomalies). Scikit-learn's implementation returns -1 for anomalies and +1 for inliers via `predict()`, and raw scores via `decision_function()` (more negative = more anomalous).

**One-Class SVM**: Learns a decision boundary in a high-dimensional feature space (via kernel trick) that encloses the "normal" data. Points outside the boundary are anomalies. Effective for small-to-medium datasets but scales poorly (O(n^2) to O(n^3) training complexity). Sensitive to kernel choice and hyperparameters.

**AutoEncoder for anomaly detection**: A neural network trained to compress input data into a low-dimensional latent representation and then reconstruct the original input. The key idea: train only on normal data, so the AutoEncoder learns to reconstruct normal patterns well. Anomalies, having different patterns, produce high reconstruction error. The reconstruction error serves as the anomaly score. PyOD's `AutoEncoder` model wraps this pattern with configurable `hidden_neurons` (e.g., `[16, 8, 8, 16]` for a 4-layer encoder-decoder) and `epochs`.

### PyOD: Unified Outlier Detection API

[PyOD](https://pyod.readthedocs.io/) is the most widely used Python library for outlier/anomaly detection, with 50+ algorithms under a consistent API. Published in [JMLR](https://www.jmlr.org/papers/volume20/19-011/19-011.pdf) (2019) and updated as [PyOD 2](https://dl.acm.org/doi/10.1145/3701716.3715196) (2025) with deep learning integration and LLM-powered model selection.

**Unified API** -- every PyOD detector follows the same pattern:

```python
from pyod.models.ecod import ECOD

detector = ECOD(contamination=0.05)
detector.fit(X_train)

scores = detector.decision_function(X_test)  # Raw anomaly scores
labels = detector.predict(X_test)             # Binary labels (0=normal, 1=outlier)
probas = detector.predict_proba(X_test)       # Probability estimates
```

**Key attributes after `fit()`**:
- `detector.decision_scores_` -- anomaly scores for the training data
- `detector.labels_` -- binary labels for the training data (based on contamination)
- `detector.threshold_` -- score threshold derived from contamination

**`contamination` parameter**: The expected proportion of outliers in the training data (default 0.1 = 10%). This is used to set the decision threshold: after fitting, the threshold is set so that `contamination * 100%` of training samples are labeled as outliers. In practice, you should set this to your best estimate of the anomaly rate. If unknown, leave at 0.1 and tune based on evaluation.

**Key algorithms used in this practice**:

| Algorithm | Type | Key Idea | Strengths |
|-----------|------|----------|-----------|
| **ECOD** (Empirical Cumulative Distribution) | Statistical | Computes tail probabilities per dimension using empirical CDF, aggregates across dimensions | Parameter-free, interpretable, fast, no assumptions about data distribution |
| **COPOD** (Copula-Based Outlier Detection) | Statistical | Models multivariate distribution using empirical copulas, estimates tail probabilities | Parameter-free, captures some cross-dimension dependencies, fast |
| **KNN** (K-Nearest Neighbors) | Distance-based | Uses distance to k-th nearest neighbor as anomaly score; farther = more anomalous | Intuitive, works well with clusters, adapts to local density |
| **AutoEncoder** | Deep Learning | Neural network trained to reconstruct input; high reconstruction error = anomaly | Captures nonlinear relationships, learns complex normal patterns |

### Evaluation Metrics for Anomaly Detection

Anomaly detection is inherently an **imbalanced classification problem** -- anomalies are rare by definition (typically 1-5% of data). This makes standard accuracy misleading: a model that labels everything as "normal" achieves 95%+ accuracy on data with 5% anomalies.

**ROC AUC (Receiver Operating Characteristic - Area Under Curve)**: Plots True Positive Rate (recall) vs. False Positive Rate across all possible thresholds. AUC = 1.0 is perfect; 0.5 is random. ROC AUC is threshold-independent -- it evaluates the detector's ability to rank anomalies higher than normal points, regardless of where you set the cutoff. Good for comparing detectors when the operating threshold has not been chosen yet.

**PR AUC (Precision-Recall - Area Under Curve)**: Plots Precision vs. Recall across all thresholds. More informative than ROC AUC when classes are highly imbalanced, because it focuses on the minority class (anomalies). A high PR AUC means the detector can identify anomalies without generating many false positives. In production alerting, this maps directly to "can we page the on-call engineer only when there is a real problem?"

**Precision, Recall, F1**: Computed at a specific threshold. Precision = "of the alerts fired, how many were real?" Recall = "of the real anomalies, how many did we catch?" F1 = harmonic mean of precision and recall. In AIOps, the tradeoff is critical: low precision means alert fatigue (too many false alarms), low recall means missed incidents.

### Where This Fits in the AIOps Ecosystem

Production AIOps platforms build on the same foundations practiced here:

- **[Datadog ML Monitors](https://www.datadoghq.com/knowledge-center/aiops/)**: Uses anomaly detection algorithms on metric streams. Their "anomaly" monitor type uses SARIMA-based models and adaptive algorithms similar to the statistical baselines in this practice.
- **Prometheus + Grafana**: Prometheus collects metrics; community projects like `prometheus-anomaly-detector` apply Isolation Forest and other ML models on Prometheus metric data.
- **PagerDuty AIOps / Event Intelligence**: Correlates alerts using ML to reduce noise and identify root causes across thousands of alerts.
- **[Splunk AIOps](https://www.splunk.com/en_us/blog/learn/aiops.html)**: Applies ML to logs and metrics for pattern detection, predictive alerting, and automated remediation.

This practice builds the core anomaly detection pipeline that sits at the heart of all these platforms: ingest metrics, establish baselines, detect deviations, evaluate detector performance.

### References

- [PyOD Documentation](https://pyod.readthedocs.io/) -- API reference, algorithm list, examples
- [PyOD GitHub](https://github.com/yzhao062/pyod) -- Source code, benchmarks, 50+ algorithm implementations
- [scikit-learn IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) -- API reference and parameter descriptions
- [Gartner AIOps Definition](https://www.gartner.com/en/information-technology/glossary/aiops-artificial-intelligence-operations) -- Original definition
- [ECOD Paper (arXiv)](https://arxiv.org/abs/2201.00382) -- "Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions"
- [COPOD Paper (arXiv)](https://arxiv.org/abs/2009.09463) -- "COPOD: Copula-Based Outlier Detection"
- [Isolation Forest (Wikipedia)](https://en.wikipedia.org/wiki/Isolation_forest) -- Algorithm overview with mathematical foundations

## Description

Build a complete anomaly detection pipeline on synthetic microservice metrics. You will generate realistic time-series data for 5 microservices (with injected spike, drift, and correlation anomalies), implement statistical baseline detectors (Z-score, moving average), train a scikit-learn Isolation Forest, compare multiple PyOD detectors (ECOD, COPOD, KNN, AutoEncoder) under a unified API, and evaluate all detectors using ROC curves, PR curves, and threshold-tuned F1 scores.

### What you'll learn

1. **Statistical baselines** -- rolling Z-score and moving average band detectors on univariate time series
2. **Isolation Forest** -- how random partitioning isolates anomalies via shorter path lengths
3. **PyOD unified API** -- fit/predict/decision_function pattern across 4+ different algorithms
4. **Evaluation framework** -- ROC AUC, PR AUC, precision/recall tradeoffs for imbalanced anomaly detection

## Instructions

### Exercise 1: Statistical Baselines (~20 min)

**File:** `src/01_statistical_baselines.py`

**What it teaches:** The foundation of all anomaly detection -- statistical methods that compare each data point to a local baseline computed from recent history. Z-score measures deviation in units of standard deviation; moving average band defines a "normal" corridor. These are the methods used by most production alerting systems (Prometheus, Datadog, CloudWatch) as their default anomaly detectors. Understanding their strengths and limitations is essential before reaching for ML-based approaches.

**What to implement:**
- `detect_zscore()` -- rolling Z-score with configurable window and threshold
- `detect_moving_average()` -- moving average band with configurable window and band width

### Exercise 2: Isolation Forest (~20 min)

**File:** `src/02_isolation_forest.py`

**What it teaches:** The transition from univariate statistical methods to multivariate ML detectors. Isolation Forest detects anomalies by exploiting the principle that unusual data points are easier to isolate via random partitioning -- they require fewer splits, resulting in shorter tree paths. Unlike statistical baselines that look at one metric at a time, Isolation Forest considers all metrics simultaneously, catching collective anomalies where no single metric exceeds its individual threshold but the combination is abnormal.

**What to implement:**
- `train_isolation_forest()` -- configure and fit sklearn's IsolationForest
- `detect_anomalies()` -- predict with the model, convert labels, extract scores

### Exercise 3: PyOD Model Comparison (~25 min)

**File:** `src/03_pyod_models.py`

**What it teaches:** PyOD's unified API lets you swap between 50+ anomaly detection algorithms with zero code changes. This exercise compares four fundamentally different approaches: ECOD (empirical distribution tails), COPOD (copula-based multivariate), KNN (distance-based), and AutoEncoder (neural network reconstruction error). The goal is to see how different algorithmic philosophies perform on the same data, building intuition for which detector to choose in production.

**What to implement:**
- `build_detector_suite()` -- instantiate PyOD detectors with appropriate parameters
- `fit_and_score()` -- use the unified fit/decision_function/predict API

### Exercise 4: Evaluation Framework (~25 min)

**File:** `src/04_evaluation.py`

**What it teaches:** In production AIOps, a detector is only as good as its evaluation. This exercise builds a comparison framework that computes ROC AUC (threshold-independent ranking quality), PR AUC (minority-class-focused performance), and threshold-specific precision/recall/F1. The ROC and PR curve plots give visual intuition for how each detector trades off false positives against missed anomalies -- the central question in alerting: "do we page the on-call or not?"

**What to implement:**
- `compute_metrics()` -- compute ROC AUC, PR AUC, precision, recall, F1 from predictions and scores
- `plot_roc_pr_curves()` -- matplotlib side-by-side ROC and PR curves for all detectors

## Motivation

- **AIOps is the future of observability**: With distributed systems growing in complexity, manual monitoring cannot scale. ML-driven anomaly detection is the foundation layer of every AIOps platform.
- **Bridges ML and DevOps skills**: This practice applies ML algorithms to operational infrastructure data -- a skill set in high demand as SRE and ML engineering roles converge.
- **Industry adoption**: Datadog, Splunk, PagerDuty, and New Relic all ship ML-based anomaly detection. Understanding the algorithms behind these tools enables better configuration and custom solutions.
- **Foundation for advanced AIOps**: This practice covers detection; follow-on practices (070b, 070c) can cover root cause analysis, correlation, and automated remediation.

## Commands

All commands run from `practice_070a_aiops_anomaly_detection/`.

### Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install Python dependencies (pyod, scikit-learn, pandas, numpy, matplotlib) |

### Exercises

| Command | Description |
|---------|-------------|
| `uv run python src/00_generate_metrics.py` | Generate synthetic metrics for 5 microservices with injected anomalies |
| `uv run python src/01_statistical_baselines.py` | Exercise 1: Run Z-score and moving average baseline detectors |
| `uv run python src/02_isolation_forest.py` | Exercise 2: Run Isolation Forest on multivariate metrics |
| `uv run python src/03_pyod_models.py` | Exercise 3: Compare ECOD, COPOD, KNN, and AutoEncoder detectors |
| `uv run python src/04_evaluation.py` | Exercise 4: Evaluate all detectors with ROC/PR curves and metrics |

### Cleanup

| Command | Description |
|---------|-------------|
| `python clean.py` | Remove all generated data, plots, models, caches, and virtual environments |

## References

- [PyOD Documentation](https://pyod.readthedocs.io/)
- [PyOD GitHub](https://github.com/yzhao062/pyod)
- [scikit-learn IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [Gartner AIOps Definition](https://www.gartner.com/en/information-technology/glossary/aiops-artificial-intelligence-operations)
- [ECOD Paper](https://arxiv.org/abs/2201.00382)
- [COPOD Paper](https://arxiv.org/abs/2009.09463)
- [Isolation Forest (Wikipedia)](https://en.wikipedia.org/wiki/Isolation_forest)
- [Datadog AIOps](https://www.datadoghq.com/knowledge-center/aiops/)
- [Splunk AIOps](https://www.splunk.com/en_us/blog/learn/aiops.html)

## State

`not-started`
