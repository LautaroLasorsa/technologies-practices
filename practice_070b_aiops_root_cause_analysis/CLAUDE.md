# Practice 070b -- AIOps: Root Cause Analysis with Service Graphs

## Technologies

- **NetworkX** -- Graph construction, traversal, and centrality algorithms (PageRank, betweenness)
- **PyOD** -- Outlier detection toolkit for computing per-service anomaly scores
- **scikit-learn** -- Preprocessing (normalization, scaling) and metrics
- **pandas** -- Time-series metric storage and correlation computation
- **NumPy** -- Numerical operations, cross-correlation, synthetic data generation
- **matplotlib** -- Graph visualization and metric plotting

## Stack

Python 3.12 (uv). No Docker needed.

## Theoretical Context

### The Root Cause Analysis Problem in Distributed Systems

When a microservice architecture experiences a performance degradation or outage, the symptoms rarely point directly to the source. A single database becoming slow can cause cascading latency increases across dozens of downstream services. An operator monitoring dashboards sees many services turning red simultaneously -- but which service *caused* the cascade, and which are merely *victims*?

**Root Cause Analysis (RCA)** is the process of tracing from observed symptoms (high latency, elevated error rates, resource exhaustion) backward through the system's dependency structure to identify the originating fault. In a monolith, RCA is relatively straightforward -- a stack trace or profiler often suffices. In microservices, the problem is fundamentally harder because:

1. **Cascading failures obscure the origin.** When service A depends on service B which depends on service C, a fault in C manifests as latency in B *and* A. By the time alerts fire, services A and B may look worse than C (amplification effect).
2. **Partial observability.** Not every service emits the same telemetry. Some have detailed tracing; others only expose coarse metrics.
3. **Non-trivial dependency topology.** Real systems have hundreds of services with complex, sometimes cyclic, dependency patterns. Manual reasoning about propagation paths is infeasible.
4. **Temporal dynamics.** Anomalies propagate with delay -- upstream latency increases appear *before* downstream ones, but the gap may be seconds to minutes depending on retry policies, circuit breakers, and queue depths.

Traditional approaches (manual triage, rule-based alerts, static runbooks) scale poorly. AIOps applies machine learning and graph analysis to automate RCA, reducing Mean Time To Resolution (MTTR) from hours to minutes.

### Service Dependency Graphs

Microservices form a **directed graph** through their interactions:

- **Nodes** represent services (API gateway, user-service, payment-service, database-proxy, etc.)
- **Directed edges** represent dependencies: an edge from A to B means "A calls B" (A depends on B). If B slows down, A is affected.
- **Edge attributes** carry operational metadata: average call latency, calls per second, error rate, protocol (HTTP, gRPC, message queue).
- **Node attributes** carry service metadata: team ownership, tier (frontend/backend/data), resource metrics (CPU, memory, latency, error rate).

In practice, dependency graphs are constructed from **distributed traces** (OpenTelemetry, Jaeger, Zipkin) that record the call chain for each request, or from **service mesh** sidecar data (Istio, Envoy). For this practice, we synthesize the graph directly.

The key insight for RCA is that the dependency graph encodes **causality direction**: if edge A -> B exists, then a fault in B can cause symptoms in A, but not vice versa (in the dependency sense). RCA therefore involves **backward traversal** -- starting from symptomatic nodes and following edges in reverse to find upstream origins.

### Anomaly Propagation Models

When a root cause fault occurs (e.g., a database runs out of connections), the anomaly **propagates** through the dependency graph:

1. **Direct impact**: The faulty service's own metrics degrade (latency spikes, error rate increases).
2. **Upstream propagation**: Services that *call* the faulty service experience increased latency (waiting for slow responses) and potentially increased errors (timeouts, retries).
3. **Amplification**: Services with high call frequency to the faulty service are affected more. Retry logic can amplify the problem (retry storms).
4. **Attenuation**: As the anomaly propagates further from the root cause, its magnitude typically decreases. Services 3 hops away see less impact than direct callers.
5. **Temporal delay**: Each hop adds latency. If service C becomes slow at time t, service B (calling C) shows degradation at t+delta1, and service A (calling B) at t+delta1+delta2.

These properties -- **attenuation with distance** and **temporal delay** -- are the key signals that RCA algorithms exploit to distinguish root causes from symptoms.

### Graph-Based RCA Approaches

#### Backward Propagation Tracing

The simplest graph-based RCA approach: given a set of anomalous nodes (services with metrics exceeding normal thresholds), trace backward through the dependency edges to find the "deepest" anomalous node -- the one with no anomalous predecessors. This is a BFS/DFS traversal on the reversed graph, starting from symptomatic leaf nodes.

**Strengths**: Simple, interpretable, fast. **Weaknesses**: Assumes a single root cause, doesn't handle cycles, doesn't account for anomaly magnitude or temporal ordering.

#### Personalized PageRank for RCA

**PageRank** ([Brin & Page, 1998](http://infolab.stanford.edu/~backrub/google.html)) was originally designed to rank web pages by importance based on link structure. The intuition: a page is important if many important pages link to it. The algorithm iteratively distributes "rank" along edges until convergence.

**Personalized PageRank (PPR)** biases the random walk toward a specific set of "seed" nodes. Instead of the random surfer restarting at any node uniformly, they restart at the seed nodes with probability proportional to the personalization vector.

**Application to RCA** ([MicroRCA, Wu et al. 2020](https://inria.hal.science/hal-02441640/file/main.pdf); [TraceRank](https://arxiv.org/html/2408.00803v1)):

1. **Construct the dependency graph** with edge weights representing call frequency or latency.
2. **Compute anomaly scores** for each node (e.g., z-score of latency relative to baseline).
3. **Build a personalization vector** where anomalous nodes receive weight proportional to their anomaly score, and normal nodes receive zero (or near-zero) weight.
4. **Run Personalized PageRank** on the graph with this personalization vector.
5. **Rank nodes by PPR score**: nodes that are both (a) upstream of many anomalous nodes and (b) themselves anomalous will receive the highest scores.

The intuition: the random walker, biased toward anomalous nodes, tends to "flow backward" through dependency edges and accumulate at the originating fault. A service that influences (is depended upon by) many anomalous downstream services naturally collects high PageRank. The `alpha` (damping factor) parameter controls how far the walk explores from seed nodes -- lower alpha means more exploration, higher alpha means staying close to seeds.

NetworkX provides `nx.pagerank(G, alpha=0.85, personalization=dict)` which directly supports this approach.

#### Betweenness Centrality for Bottleneck Detection

**Betweenness centrality** measures how often a node appears on shortest paths between other nodes. A node with high betweenness is a "bottleneck" or "bridge" -- many communication paths flow through it.

For RCA, the insight is: among the set of anomalous nodes, those with high betweenness centrality in the anomalous subgraph are likely to be propagation hubs -- either root causes or critical intermediaries. If you extract the subgraph induced by anomalous nodes and their immediate neighbors, betweenness centrality identifies which nodes sit on the most propagation paths.

**Weighted betweenness** (using call frequency as edge weight) further refines this: high-traffic bottleneck nodes are more likely to be root causes. NetworkX provides `nx.betweenness_centrality(G, weight='weight')`.

**Combined ranking**: In practice, Personalized PageRank and betweenness centrality capture complementary signals. PPR captures "upstream influence on anomalous nodes" while betweenness captures "structural bottleneck position." A weighted combination of both often outperforms either alone.

### Correlation-Based Causal Inference

Graph topology alone doesn't capture all causal information. **Metric correlation** between services provides additional evidence:

#### Pearson Correlation

If two services are causally linked (one's fault causes the other's degradation), their anomaly metrics will be **correlated over time**. Computing pairwise Pearson correlation between service latency (or error rate) time series reveals which services are "co-anomalous."

**Limitation**: Correlation does not imply causation. Two services may be correlated because they share a common upstream dependency, not because one causes the other's anomaly. This is where temporal information becomes essential.

#### Granger Causality

**Granger causality** ([Granger, 1969](https://en.wikipedia.org/wiki/Granger_causality)) tests whether one time series helps predict another. Formally: X Granger-causes Y if past values of X improve the prediction of Y beyond what past values of Y alone provide.

For RCA, this translates to: if service A's anomaly **precedes** service B's anomaly consistently, A may be the cause. Specifically:

1. Compute **cross-correlation** between anomaly indicator time series of services A and B at various lags.
2. If peak cross-correlation occurs at a **positive lag** (A leads B), A may cause B's anomaly.
3. The **lag value** should be consistent with the known network latency between A and B.

**CloudRanger** ([Wang et al., 2018](https://github.com/dreamhomes/RCAPapers)) uses Pearson correlation to construct a service causal graph, then performs a second-order random walk to localize root causes. **RUN** ([Lin et al., AAAI 2024](https://arxiv.org/abs/2402.01140)) extends this with neural Granger causal discovery using contrastive learning to capture non-linear temporal dependencies.

A practical simplification (used in this practice): compute cross-correlation at discrete lags and check whether the source service's anomaly indicators consistently peak *before* the target service's. This provides a "temporal precedence score" that, combined with Pearson correlation magnitude, yields a lightweight causal inference signal.

### Research Landscape: MicroRCA, CloudRanger, and Beyond

| System | Method | Key Innovation |
|--------|--------|----------------|
| **MicroRCA** (Wu et al., 2020) | Attributed graph + Personalized PageRank random walk | Combines service metrics + host metrics into a single attributed graph; uses anomaly score as PPR personalization |
| **CloudRanger** (Wang et al., 2018) | Pearson correlation causal graph + second-order random walk | Dynamically constructs impact graph from metric correlations; no dependency graph needed |
| **TraceRank** (Yu et al., 2021) | Trace-based PageRank | Uses distributed traces (not just metrics) to build dependency graph; PageRank-inspired walk |
| **Groot** (Wang et al., 2021) | Event causality graph + GrootRank | Constructs causal graphs from events (logs, alerts); custom PageRank variant (GrootRank) |
| **RUN** (Lin et al., AAAI 2024) | Neural Granger causal discovery + contrastive learning | Captures non-linear temporal causality; state-of-the-art on standard benchmarks |
| **ClearCausal** (2024) | Cross-layer causal analysis | Fuses application-layer and infrastructure-layer metrics for cross-layer RCA |

Common themes across all approaches:
- **Graph structure matters**: whether from traces, metrics, or static config, encoding service dependencies as a graph is fundamental
- **Anomaly scores as weights**: every approach assigns per-node anomaly scores that bias the ranking algorithm
- **Random walk / PageRank**: the dominant algorithmic primitive for converting graph structure + anomaly scores into a root cause ranking
- **Temporal information improves accuracy**: methods that incorporate "which anomaly came first" consistently outperform those that only look at anomaly magnitude

### What This Practice Covers

This practice implements three complementary RCA techniques:

1. **Anomaly propagation tracing** -- BFS/DFS backward through the dependency graph to find propagation chains from symptoms to potential root causes
2. **Centrality-based ranking** -- Personalized PageRank + betweenness centrality to rank candidate root causes
3. **Correlation-based causal inference** -- Pearson correlation + temporal precedence (simplified Granger) to infer causal relationships from metric time series

Together, these represent the core toolkit used in production AIOps systems. The synthetic data generator creates a realistic cascading failure scenario with known ground truth, allowing you to evaluate each technique's accuracy.

## Description

Build a complete root cause analysis pipeline for a simulated microservice system. A synthetic service dependency graph (15-20 nodes) is generated with realistic metrics. A cascading failure is injected from known root cause nodes, propagating through dependency edges with temporal delay and attenuation. You then implement three RCA techniques to recover the true root causes from the observed symptoms.

### What you'll learn

1. **Service dependency graph construction** -- modeling microservices as directed graphs with operational metadata
2. **Anomaly propagation mechanics** -- how faults cascade through dependencies with delay and attenuation
3. **Backward propagation tracing** -- BFS/DFS on reversed graphs to find propagation chains
4. **Personalized PageRank for RCA** -- biasing random walks toward anomalous nodes to rank root causes
5. **Betweenness centrality** -- identifying structural bottleneck nodes in the anomalous subgraph
6. **Correlation-based causal inference** -- using Pearson correlation and temporal cross-correlation to infer causality from metrics
7. **Combined ranking strategies** -- merging multiple signals (topology, centrality, correlation) for robust RCA

## Instructions

### Phase 1: Setup & Topology Generation (~10 min)

1. Install dependencies with `uv sync`
2. Run `uv run python src/00_build_topology.py` to generate the service dependency graph, synthetic metrics, and ground truth
3. Inspect the outputs in `data/`: `service_graph.json` (adjacency data), `metrics.csv` (time-series per service), `ground_truth.json` (true root causes)
4. Look at the generated `plots/topology.png` to understand the dependency structure
5. **Key question**: In the generated graph, if a "data tier" service becomes slow, which "frontend tier" services would be affected? Trace the dependency edges mentally.

### Phase 2: Anomaly Propagation Tracing (~25 min)

1. Review `src/01_anomaly_propagation.py` -- understand how anomaly scores are loaded and the scaffolded visualization code
2. **Exercise 1a** -- Implement `trace_anomaly_propagation()`: Given the DAG and per-node anomaly scores, trace backward from highly-anomalous leaf nodes through predecessor edges to build propagation chains. This teaches how graph traversal can recover the "story" of a cascading failure -- which service infected which. The key decision: when do you stop tracing backward?
3. **Exercise 1b** -- Implement `score_propagation_likelihood()`: Score each chain by how plausible it is as a real cascade. This teaches that not all paths through the graph are equally likely -- temporal ordering and edge traffic volume matter.
4. Run `uv run python src/01_anomaly_propagation.py` and compare the top chains with ground truth

### Phase 3: Centrality-Based Root Cause Ranking (~25 min)

1. Review `src/02_root_cause_ranking.py` -- understand the ranking framework and visualization scaffolding
2. **Exercise 2a** -- Implement `rank_by_pagerank()`: Use NetworkX's Personalized PageRank with a personalization vector biased toward anomalous nodes. This teaches the core MicroRCA/TraceRank algorithm -- how a biased random walk on a dependency graph naturally converges on root cause nodes.
3. **Exercise 2b** -- Implement `rank_by_centrality()`: Compute betweenness centrality on the anomalous subgraph. This teaches how structural position (bottleneck vs. leaf) provides complementary RCA signal.
4. **Exercise 2c** -- Implement `combined_ranking()`: Merge PageRank and centrality scores into a final ranking. This teaches the common AIOps pattern of fusing multiple signals.
5. Run `uv run python src/02_root_cause_ranking.py` and evaluate precision/recall against ground truth

### Phase 4: Correlation-Based Causal Inference (~25 min)

1. Review `src/03_correlation_rca.py` -- understand how metrics are loaded and the causal graph visualization
2. **Exercise 3a** -- Implement `compute_pairwise_correlation()`: Compute Pearson correlation matrix between service metrics. This teaches how metric co-movement reveals causal relationships -- but also why correlation alone is insufficient (confounding).
3. **Exercise 3b** -- Implement `temporal_precedence_score()`: Compute a simplified Granger-like temporal precedence score using cross-correlation at various lags. This teaches the critical insight that **temporal ordering** distinguishes cause from effect -- the root cause's anomaly appears *before* the victim's.
4. Run `uv run python src/03_correlation_rca.py` and compare the inferred causal graph with the true dependency graph

## Motivation

- **Critical SRE/DevOps skill**: Root cause analysis is the most time-consuming activity during incidents. Automating RCA with graph analysis directly reduces MTTR.
- **Graph algorithms in practice**: PageRank and centrality measures are standard tools -- applying them to RCA demonstrates practical graph algorithm fluency beyond textbook examples.
- **Bridges ML and systems engineering**: Combines anomaly detection (ML) with graph theory and causal inference -- the exact intersection where AIOps operates.
- **Production relevance**: MicroRCA, CloudRanger, and similar systems are deployed at Google, Microsoft, Netflix, and Uber. Understanding the algorithmic foundations is essential for building or evaluating AIOps tools.
- **Complements 070a**: Builds on anomaly detection concepts by adding the graph-based reasoning layer that turns "which services are anomalous?" into "which service *caused* the anomaly?"

## Commands

All commands run from `practice_070b_aiops_root_cause_analysis/`.

### Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install all Python dependencies |

### Running Exercises

| Command | Description |
|---------|-------------|
| `uv run python src/00_build_topology.py` | Build service dependency graph, generate synthetic metrics with injected cascading failure, save to data/ and plots/ |
| `uv run python src/01_anomaly_propagation.py` | Exercise 1: Trace anomaly propagation backward through dependency edges |
| `uv run python src/02_root_cause_ranking.py` | Exercise 2: Rank root cause candidates using PageRank + betweenness centrality |
| `uv run python src/03_correlation_rca.py` | Exercise 3: Infer causal relationships using Pearson correlation + temporal precedence |

### Cleanup

| Command | Description |
|---------|-------------|
| `python clean.py` | Remove all generated data, plots, models, and caches |

## References

- [MicroRCA: Root Cause Localization of Performance Issues in Microservices (Wu et al., 2020)](https://inria.hal.science/hal-02441640/file/main.pdf)
- [CloudRanger: Root Cause Identification for Cloud Native Systems (Wang et al., 2018)](https://github.com/dreamhomes/RCAPapers)
- [Root Cause Analysis in Microservice Using Neural Granger Causal Discovery (Lin et al., AAAI 2024)](https://arxiv.org/abs/2402.01140)
- [Groot: An Event-graph-based Approach for Root Cause Analysis (Wang et al., 2021)](https://arxiv.org/pdf/2108.00344)
- [A Comprehensive Survey on Root Cause Analysis in (Micro) Services (2024)](https://arxiv.org/html/2408.00803v1)
- [NetworkX: Centrality Algorithms](https://networkx.org/documentation/stable/reference/algorithms/centrality.html)
- [NetworkX: pagerank](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.pagerank.html)
- [NetworkX: betweenness_centrality](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.betweenness_centrality.html)
- [Granger Causality (Wikipedia)](https://en.wikipedia.org/wiki/Granger_causality)
- [PyOD: Python Outlier Detection](https://pyod.readthedocs.io/en/latest/)

## State

`not-started`
