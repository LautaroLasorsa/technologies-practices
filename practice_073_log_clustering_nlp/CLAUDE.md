# Practice 073 -- Log Clustering & Analysis with NLP

## Technologies

- **drain3** -- Streaming log template miner based on the Drain algorithm
- **sentence-transformers** -- Pre-trained models for dense text embeddings (all-MiniLM-L6-v2)
- **scikit-learn** -- KMeans clustering, IsolationForest anomaly detection, silhouette scoring
- **HDBSCAN** -- Density-based clustering that automatically determines cluster count
- **UMAP** -- Dimensionality reduction for embedding visualization
- **pandas / numpy** -- Data manipulation and numerical operations
- **matplotlib** -- Plotting and visualization

## Stack

Python 3.12+ (uv). No Docker needed.

## Theoretical Context

### The Log Analysis Problem

Distributed systems produce millions of log lines per day across dozens of services. A moderately-sized Kubernetes cluster running 30 microservices, each logging at 100 lines/second, generates **260 million log lines per day**. Manual inspection is impossible. Traditional approaches -- grep, regex dashboards, keyword alerts -- break down because:

1. **Volume**: No human can scan millions of lines. Keyword searches miss novel failure modes.
2. **Semi-structured format**: Logs are a mix of fixed templates and variable parameters. The line `"Connection to 192.168.1.5:3306 timed out after 30s"` contains a fixed template (`"Connection to <IP>:<PORT> timed out after <NUM>s"`) and variable parts (the specific IP, port, and duration). Treating the whole line as unique text produces millions of "distinct" messages that are really just a few dozen templates with different parameters.
3. **Cross-service correlation**: An anomaly in one service manifests as cascading log patterns across dependent services. Understanding these patterns requires grouping and correlating logs semantically, not just lexically.

The solution pipeline addressed in this practice:

```
Raw logs --> Template extraction (Drain) --> Semantic embedding --> Clustering --> Anomaly detection
```

Each stage reduces the dimensionality of the problem: millions of raw lines become hundreds of templates, which cluster into tens of semantic groups, whose frequency patterns reveal anomalies.

### Log Parsing: From Raw Lines to Structured Templates

**Log parsing** converts free-text log lines into structured `(template, parameters)` pairs. For example:

| Raw log line | Template | Parameters |
|---|---|---|
| `Connection to 192.168.1.5:3306 timed out after 30s` | `Connection to <IP>:<PORT> timed out after <NUM>s` | `192.168.1.5`, `3306`, `30` |
| `User alice logged in from 10.0.0.1` | `User <*> logged in from <IP>` | `alice`, `10.0.0.1` |
| `User bob logged in from 10.0.0.2` | `User <*> logged in from <IP>` | `bob`, `10.0.0.2` |

The two "User logged in" lines map to the **same template** despite different parameter values. This is the key insight: the template captures the *event type*, while parameters capture the *instance details*.

**Why not regex?** Manually writing regex patterns for every log format across dozens of services is impractical. Patterns break when developers change log messages. New services require new patterns. Automated log parsing discovers templates from the data itself.

### The Drain Algorithm

Drain (He et al., 2017) is the state-of-the-art online log parser. "Online" means it processes logs one line at a time in a streaming fashion -- no need to batch the entire dataset. It achieves **O(1) amortized time per log line** by using a fixed-depth parse tree.

**How the parse tree works:**

```
                        Root
                         |
            +-----------+-----------+
            |           |           |
         Length=8    Length=10    Length=12    <-- Level 1: log message length (token count)
            |           |
         +--+--+     +--+--+
         |     |     |     |
      "User" "Conn" "GET" "POST"              <-- Level 2: first token of the message
         |     |
       [log   [log                             <-- Leaf nodes: lists of log groups (clusters)
       groups] groups]
```

**Processing a new log line:**

1. **Tokenize** the log line by whitespace (and any configured extra delimiters).
2. **Level 1 -- Length**: Route to the child node matching the token count. If the line has 10 tokens, go to the "Length=10" node.
3. **Level 2 -- First token**: Route to the child matching the first token. If the line starts with "Connection", go to the "Connection" node.
4. **Leaf node -- Similarity match**: Compare the log line against existing templates in the leaf node using token-level similarity. If similarity exceeds a threshold (`sim_th`), merge with that template (replacing differing tokens with `<*>`). Otherwise, create a new template.

**Key parameters:**
- `sim_th` (similarity threshold, default 0.4): Fraction of tokens that must match for a log line to join an existing cluster. Too high (>0.7) causes over-splitting (many nearly-identical templates). Too low (<0.3) causes under-splitting (unrelated logs merge into one template).
- `depth` (default 4): Tree depth. Deeper trees are more selective but use more memory. Minimum is 3 (root + length + first-token + leaf).
- `max_children` (default 100): Maximum children per internal node. Limits memory usage.

**Why Drain beats regex:**
- **No manual pattern writing**: Templates emerge automatically from the data.
- **Adapts to new log formats**: New services' logs are parsed without configuration changes.
- **Fast**: Fixed-depth tree means constant-time lookup regardless of how many templates exist.
- **Online**: Processes logs as they arrive, suitable for real-time pipelines.

**Original paper**: [He, P., Zhu, J., Zheng, Z., & Lyu, M. R. (2017). "Drain: An Online Log Parsing Approach with Fixed Depth Tree." ICWS 2017.](https://jiemingzhu.github.io/pub/pjhe_icws2017.pdf)

### drain3: Python Implementation

[drain3](https://github.com/logpai/Drain3) is the production-quality Python implementation of Drain maintained by the LogPAI team. Key features:

- **Configurable masking**: Before template mining, drain3 can replace known variable patterns (IP addresses, UUIDs, file paths, numbers) with named placeholders using regex. This improves template quality -- without masking, `192.168.1.5` and `10.0.0.1` would be treated as distinct tokens and force separate templates.
- **Persistence**: Save/load miner state (learned templates) to files, Redis, or Kafka for production streaming.
- **Snapshot/restore**: Periodically snapshot the parse tree for fault tolerance.
- **Template matching**: `match()` method classifies new logs against learned templates without creating new clusters (inference mode).
- **Parameter extraction**: `extract_parameters()` returns the variable values from a log line given its template.

**Core API:**

```python
from drain3.template_miner import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

config = TemplateMinerConfig()
config.drain_sim_th = 0.4
config.drain_depth = 4
config.masking_instructions = [...]  # regex masking rules

miner = TemplateMiner(config=config)

# Process a log line (returns dict with cluster_id, template_mined, change_type)
result = miner.add_log_message("Connection to 10.0.0.1:3306 timed out after 30s")
# result["template_mined"] = "Connection to <IP>:<PORT> timed out after <NUM>s"

# Classify without learning
cluster = miner.match("Connection to 10.0.0.2:5432 timed out after 15s")
```

**Masking configuration** uses `RegexMaskingInstruction` objects or JSON in a `drain3.ini` file:

```json
[
  {"regex_pattern": "(\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3})", "mask_with": "IP"},
  {"regex_pattern": "([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", "mask_with": "UUID"},
  {"regex_pattern": "((?<=[^A-Za-z0-9])|^)([\\-\\+]?\\d+)((?=[^A-Za-z0-9])|$)", "mask_with": "NUM"}
]
```

Parameters not matching any mask are replaced with `<*>` (the wildcard placeholder).

### Log Embeddings with Sentence-Transformers

Once logs are parsed into templates, we need to measure **semantic similarity** between templates. String comparison fails here: `"Connection to <IP>:<PORT> timed out"` and `"Socket read from <IP> timed out after <NUM>s"` describe similar failures but share few tokens.

**Sentence-transformers** ([sbert.net](https://www.sbert.net/)) provide pre-trained models that encode text into dense vector embeddings where semantically similar texts have high cosine similarity.

**all-MiniLM-L6-v2** is the recommended model for this practice:
- **384-dimensional** embeddings (compact, fast)
- **22 MB** model size (runs on CPU in seconds)
- Trained on 1 billion sentence pairs with contrastive learning
- Excellent for short texts like log templates (optimized for sentences, not documents)

**Usage:**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(["Connection timed out", "Socket read timeout"])
# embeddings.shape = (2, 384)
# cosine_similarity(embeddings[0], embeddings[1]) ≈ 0.85
```

**Why embed log templates (not raw logs)?** Raw logs have variable parts (IPs, timestamps) that add noise. Templates isolate the semantic event type, producing cleaner, more meaningful embeddings. Embedding 200 unique templates is also orders of magnitude faster than embedding 10,000 raw log lines.

### Clustering: KMeans vs HDBSCAN

With template embeddings in hand, clustering groups semantically related templates into categories (e.g., "network errors", "authentication events", "database queries").

**KMeans** ([sklearn.cluster.KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)):
- Requires specifying `K` (number of clusters) upfront.
- Assigns every point to exactly one cluster (no noise concept).
- Works well when clusters are roughly spherical and equally sized.
- **Choosing K**: Use the silhouette score -- for each `k` in a range, compute the mean silhouette coefficient (measures how similar each point is to its own cluster vs. nearest neighbor cluster). Higher is better, range [-1, 1].

**HDBSCAN** ([Hierarchical Density-Based Spatial Clustering](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html)):
- **Does not require K** -- finds the number of clusters automatically from the data density.
- Identifies **noise points** (label = -1): data points that don't belong to any dense cluster.
- Handles clusters of varying density and irregular shape.
- Key parameter: `min_cluster_size` -- minimum number of points to form a cluster. Smaller values find more clusters (including small ones); larger values require denser, bigger groups.
- Built on the idea of constructing a hierarchy of clusters at all density levels and selecting the most persistent clusters (highest "stability").

**When to use each:**
- **KMeans**: When you have domain knowledge about the expected number of categories, or when you need every point assigned.
- **HDBSCAN**: When the number of categories is unknown, when outlier detection matters, or when cluster densities vary.

For log analysis, HDBSCAN is typically preferred because the number of log categories is unknown a priori and some templates may be genuinely unique (noise).

### UMAP: Visualizing High-Dimensional Embeddings

[UMAP](https://umap-learn.readthedocs.io/en/latest/) (Uniform Manifold Approximation and Projection) reduces high-dimensional embeddings (384 dims) to 2D or 3D for visualization. It preserves both **local neighborhood structure** (similar templates stay close) and **global topology** (distinct clusters remain separated) -- a significant advantage over t-SNE, which often distorts global distances.

**Key parameters:**
- `n_neighbors` (default 15): Controls the balance between local and global structure. Low values (5-10) emphasize fine-grained local neighborhoods; high values (30-50) reveal broader patterns.
- `min_dist` (default 0.1): Controls how tightly points cluster in the output. Low values (0.0-0.05) create tight, clumpy clusters; higher values (0.3-0.5) spread points more evenly.
- `metric` (default "euclidean"): Distance metric. Use "cosine" when working with normalized embeddings.

**Practical tip**: For log template visualization, start with `n_neighbors=15, min_dist=0.05, metric="cosine"`. This produces tight clusters that are easy to visually inspect.

**Docs**: [UMAP Parameters Guide](https://umap-learn.readthedocs.io/en/latest/parameters.html)

### Log Anomaly Detection via Cluster Frequencies

After clustering templates into semantic groups, the final step is detecting **anomalous time periods** based on how cluster frequencies change over time.

**The approach:**

1. **Build frequency vectors**: Divide the log timeline into fixed windows (e.g., 5-minute buckets). For each window, count how many log lines belong to each cluster. This produces a matrix: rows = time windows, columns = cluster IDs, values = counts.
2. **Detect anomalous windows**: Use an unsupervised anomaly detector on the frequency vectors. A window is anomalous if its cluster distribution is unusual -- e.g., a sudden spike in error clusters, disappearance of normal heartbeat logs, or appearance of never-before-seen templates.

**IsolationForest** ([sklearn.ensemble.IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)) works by randomly partitioning the feature space with decision trees. Anomalous points (unusual frequency distributions) are **isolated in fewer splits** because they lie in sparse regions. The `contamination` parameter sets the expected proportion of anomalies (e.g., 0.05 = expect 5% of windows to be anomalous).

**Why this works for log anomaly detection:**
- Normal operations produce a **stable cluster distribution** -- roughly the same proportion of INFO, WARN, and ERROR templates in each window.
- Incidents change the distribution: error clusters spike, normal clusters drop, new templates appear.
- IsolationForest captures these distributional shifts without needing labeled training data.

### Real-World Tools & Ecosystem

Production log analysis tools implement variations of this pipeline:

| Tool | Approach |
|------|----------|
| **Elastic ML** | Categorization jobs that use a modified Drain-like parser + anomaly detection on category counts |
| **Splunk ITSI** | Event analytics with pattern detection and adaptive thresholding |
| **Datadog Log Patterns** | Automatic template extraction with clustering for log exploration |
| **Grafana Loki** | LogQL pattern matching + alerting on label cardinality changes |
| **AWS CloudWatch Anomaly Detection** | Statistical models on metric time series derived from log groups |

This practice teaches the foundational algorithms underlying all these commercial tools.

### References

- [He, P. et al. (2017). "Drain: An Online Log Parsing Approach with Fixed Depth Tree." ICWS 2017.](https://jiemingzhu.github.io/pub/pjhe_icws2017.pdf)
- [drain3 -- GitHub (LogPAI)](https://github.com/logpai/Drain3)
- [Sentence-Transformers Documentation](https://www.sbert.net/)
- [all-MiniLM-L6-v2 -- Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [HDBSCAN Documentation](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html)
- [UMAP Parameters Guide](https://umap-learn.readthedocs.io/en/latest/parameters.html)
- [scikit-learn IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)

## Description

Build a complete log analysis pipeline that takes raw distributed-system logs through template extraction, semantic embedding, clustering, and anomaly detection. You'll work with synthetic logs from a 5-microservice system, extract templates with Drain3, embed them for semantic similarity, cluster with both KMeans and HDBSCAN, and detect anomalous time windows using IsolationForest on cluster frequency vectors.

### What you'll learn

1. **Log template extraction** -- Configuring Drain3 masking and similarity thresholds for distributed system logs
2. **Semantic embeddings** -- Using sentence-transformers to capture meaning beyond string matching
3. **Clustering comparison** -- Understanding KMeans vs HDBSCAN trade-offs on real embedding data
4. **Dimensionality reduction** -- UMAP for visual inspection of high-dimensional clusters
5. **Frequency-based anomaly detection** -- Detecting incidents from cluster distribution shifts over time

## Instructions

### Phase 1: Setup & Data Generation (~10 min)

1. Install dependencies with `uv sync`
2. Run `uv run python src/00_generate_logs.py` to generate 10,000 synthetic log lines from 5 microservices. This creates `data/logs.txt` (raw logs) and `data/logs_metadata.csv` (with ground truth labels including an injected anomaly burst).
3. Open `data/logs.txt` and scan a few hundred lines to get a feel for the log format, the services, and the mix of INFO/WARN/ERROR messages. Notice how variable parts (IPs, request IDs, durations) make each line unique even though the underlying event type repeats.

### Phase 2: Drain3 Template Extraction (~25 min)

1. Open `src/01_drain_parsing.py` and review the scaffolded structure.
2. **Exercise 1 -- `configure_drain()`**: Configure a drain3 TemplateMiner with masking rules for distributed system logs. This teaches how automated log parsing works -- you'll define regex masks for IPs, UUIDs, numbers, and paths, then set the similarity threshold that controls template granularity. Getting masking right is critical: without it, every unique IP or request ID creates a separate template, defeating the purpose of parsing. The `sim_th` parameter is the main knob -- experiment with values between 0.3 and 0.6 to see how it affects template count.
3. **Exercise 2 -- `parse_logs()`**: Feed log lines through the TemplateMiner one at a time. This teaches the streaming nature of Drain -- each line is processed in O(1) time, and the template set evolves as new patterns are encountered. You'll collect the cluster assignment for each line.
4. Run `uv run python src/01_drain_parsing.py` and examine the output. You should see 40-80 unique templates from the 10,000 log lines. If you see hundreds of templates, your masking is too loose; if fewer than 20, your `sim_th` is too low.

### Phase 3: Semantic Embeddings (~20 min)

1. Open `src/02_embeddings.py` and review the scaffolded structure.
2. **Exercise 3 -- `embed_templates()`**: Load a sentence-transformer model and encode the unique templates into dense vectors. This teaches how NLP embeddings capture semantic meaning -- templates about timeouts will be near each other in vector space even if they use different words. The first run downloads the model (~22 MB).
3. **Exercise 4 -- `compute_similarity_matrix()`**: Compute pairwise cosine similarity between all template embeddings. This gives you a concrete view of which templates the model considers semantically related, and whether the embeddings are capturing meaningful log categories.
4. Run `uv run python src/02_embeddings.py` and inspect the top-5 most similar template pairs. Do they make semantic sense?

### Phase 4: Clustering & Visualization (~25 min)

1. Open `src/03_clustering.py` and review the scaffolded structure.
2. **Exercise 5 -- `cluster_kmeans()`**: Cluster template embeddings with KMeans. Use silhouette scoring to find the optimal K. This teaches the trade-off of KMeans: you get clean, exhaustive assignments but must guess the number of clusters.
3. **Exercise 6 -- `cluster_hdbscan()`**: Cluster with HDBSCAN. This teaches density-based clustering: it automatically finds K, identifies noise, and handles varying cluster densities. Compare the cluster count and quality with KMeans.
4. **Exercise 7 -- `visualize_clusters()`**: Use UMAP to reduce 384-dim embeddings to 2D and create scatter plots colored by cluster. This teaches how dimensionality reduction enables visual inspection of clustering quality.
5. Run `uv run python src/03_clustering.py` and compare the two visualizations in `plots/`.

### Phase 5: Anomaly Detection (~20 min)

1. Open `src/04_anomaly_detection.py` and review the scaffolded structure.
2. **Exercise 8 -- `build_frequency_vectors()`**: Group parsed logs into time windows and count cluster occurrences. This teaches the representation that makes anomaly detection possible: converting a stream of log events into a time series of frequency vectors.
3. **Exercise 9 -- `detect_anomalous_windows()`**: Apply IsolationForest to the frequency vectors. This teaches unsupervised anomaly detection on multivariate time series -- the model learns "normal" cluster distributions and flags windows that deviate.
4. Run `uv run python src/04_anomaly_detection.py` and check whether the detected anomalous windows align with the injected anomaly burst in the synthetic data.

## Motivation

- **Production observability**: Every production system needs log analysis. Understanding the algorithms behind tools like Datadog Log Patterns, Elastic ML, and Splunk ITSI makes you a more effective user and debugger of these systems.
- **ML for systems**: Log clustering sits at the intersection of NLP and distributed systems -- applying ML to operational data is a growing specialization (AIOps) with high demand.
- **Transferable techniques**: Template extraction, semantic embeddings, and frequency-based anomaly detection apply far beyond logs -- to error categorization, alert deduplication, incident triage, and root cause analysis.
- **Complements practices 007a/007b (OpenTelemetry)**: Structured telemetry and automated log analysis are two sides of the observability coin.

## Commands

All commands run from `practice_073_log_clustering_nlp/`.

### Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install all dependencies from pyproject.toml |

### Data Generation

| Command | Description |
|---------|-------------|
| `uv run python src/00_generate_logs.py` | Generate 10,000 synthetic log lines with injected anomaly burst |

### Exercises

| Command | Description |
|---------|-------------|
| `uv run python src/01_drain_parsing.py` | Exercise 1-2: Parse logs with Drain3 template extraction |
| `uv run python src/02_embeddings.py` | Exercise 3-4: Generate sentence-transformer embeddings for templates |
| `uv run python src/03_clustering.py` | Exercise 5-7: Cluster templates with KMeans/HDBSCAN and visualize with UMAP |
| `uv run python src/04_anomaly_detection.py` | Exercise 8-9: Detect anomalous time windows from cluster frequency patterns |

### Cleanup

| Command | Description |
|---------|-------------|
| `python clean.py` | Remove all generated data, plots, models, and caches |

## State

`not-started`
