# Practice 074: ML-guided Chaos Engineering

## Technologies

- **networkx** -- Graph modeling library for representing microservice dependency topologies
- **scikit-learn** -- ML framework for training blast radius prediction models (RandomForest)
- **numpy** -- Numerical computing for Monte Carlo simulations and statistical analysis
- **pandas** -- Tabular data manipulation for metrics, features, and experiment results
- **matplotlib** -- Visualization of system graphs, blast radius distributions, and experiment rankings

## Stack

- Python 3.12+ (uv)
- No Docker needed -- everything runs locally with synthetic data

## Theoretical Context

### Chaos Engineering: Origins and Principles

Chaos Engineering is the discipline of experimenting on a distributed system to build confidence in its ability to withstand turbulent conditions in production. The field originated at Netflix in 2010-2011, when engineers created **Chaos Monkey** -- a tool that randomly terminated virtual machine instances in production -- to force themselves to build resilient services after migrating from monolithic data centers to AWS. The reasoning was simple: if your system cannot survive a single instance going down, you will discover that at 3 AM during peak traffic. Better to discover it deliberately at 2 PM on a Tuesday.

Netflix expanded Chaos Monkey into the **Simian Army**: Chaos Gorilla (simulates entire AWS availability zone failure), Latency Monkey (injects artificial delays), Conformity Monkey (detects instances not following best practices), and others. This formalized into the **Principles of Chaos Engineering** (principlesofchaos.org), which define five core tenets:

1. **Build a hypothesis around steady state behavior.** Define what "normal" looks like using measurable outputs (request throughput, error rates, latency percentiles), not internal system state. The hypothesis is: "steady state will continue even when we inject failures."
2. **Vary real-world events.** Inject failures that actually happen: server crashes, network partitions, disk full, certificate expiration, clock skew, DNS failures. Prioritize by frequency and impact.
3. **Run experiments in production.** Staging environments cannot replicate the full complexity of production traffic patterns, data distributions, and emergent behaviors. The highest confidence comes from production experiments.
4. **Automate experiments to run continuously.** Manual one-off tests provide snapshots. Automated, continuous experiments catch regressions and emergent fragilities.
5. **Minimize blast radius.** Start small. The goal is learning, not outages. Use progressive widening: start with one instance, then one availability zone, then a full region.

Sources: [Principles of Chaos Engineering](https://principlesofchaos.org/), [Netflix Chaos Monkey](https://netflix.github.io/chaosmonkey/), [Gremlin: The Origin of Chaos Monkey](https://www.gremlin.com/chaos-monkey/the-origin-of-chaos-monkey)

### The Chaos Engineering Process

A chaos experiment follows a structured cycle:

1. **Define steady state.** Choose metrics that represent normal system behavior. For an e-commerce system: orders per second, P99 latency, error rate, cart abandonment rate. Steady state is defined as these metrics staying within historical bounds (e.g., mean +/- 2 standard deviations over a rolling window).

2. **Formulate a hypothesis.** "When we kill the recommendation service, the product page will still load within 500ms and show a static fallback instead of personalized recommendations." The hypothesis should be specific and falsifiable.

3. **Inject a failure.** Types of failures (detailed in next section) range from killing a single process to simulating a full network partition. The injection should be controlled and reversible.

4. **Observe the system.** Monitor whether steady state is maintained. Collect metrics, traces, and logs. Compare experimental metrics against the baseline. Key question: did the system degrade gracefully, or did it cascade into failure?

5. **Learn and improve.** If the hypothesis held, confidence increases. If it failed, you found a real vulnerability before your users did. Fix the issue, add monitors, and run the experiment again to verify the fix.

Sources: [Google Cloud: Getting started with Chaos Engineering](https://cloud.google.com/blog/products/devops-sre/getting-started-with-chaos-engineering), [IBM: What is Chaos Engineering](https://www.ibm.com/think/topics/chaos-engineering)

### Types of Failures

Chaos experiments inject failures at different layers of the stack:

| Failure Type | Examples | What It Tests |
|---|---|---|
| **Node/Process failure** | Kill a container, terminate a VM, crash a JVM | Redundancy, load balancing, health checks, auto-restart |
| **Network failure** | Partition between services, drop packets, add 500ms latency, corrupt DNS | Timeouts, retries, circuit breakers, fallback logic |
| **Resource exhaustion** | Fill disk to 100%, consume all CPU, exhaust file descriptors, OOM | Resource limits, autoscaling, graceful degradation, backpressure |
| **Dependency failure** | Database goes read-only, cache returns errors, third-party API times out | Circuit breakers, fallback data, bulkheads, stale-cache policies |
| **State/Data failure** | Clock skew between nodes, stale cache entries, corrupted messages | Idempotency, data validation, conflict resolution, event ordering |

Each failure type tests a different dimension of resilience. A system that survives node failures may completely collapse under network partitions because its retry logic causes a thundering herd.

Sources: [Steadybit: Chaos Engineering Experiments](https://steadybit.com/blog/chaos-experiments/), [Educative: Chaos Engineering 101](https://www.educative.io/blog/chaos-engineering-process-principles)

### Why Random Chaos is Inefficient

The original Chaos Monkey approach -- randomly killing instances -- was groundbreaking but inefficient. In a system with 500 microservices, randomly choosing which one to kill means most experiments will target low-impact, well-resilient services. You learn nothing new from killing a stateless frontend that has 10 replicas behind a load balancer.

The high-value experiments are the ones that target **critical failure points**: the single-replica payment service, the database that 30 other services depend on, the authentication gateway through which all traffic flows. But identifying these critical points manually requires deep architectural knowledge that becomes stale as the system evolves.

This is where **ML-guided chaos engineering** enters: instead of random selection, use the system's dependency graph and historical data to **predict which experiments will have the highest impact** and prioritize those. The goal is not to maximize damage -- it is to maximize learning per experiment.

### Blast Radius: Measuring Failure Impact

**Blast radius** is the scope of impact when a component fails. It answers: "If this service goes down, how many other services are affected, and how severely?"

Blast radius depends on several factors:
- **Number of dependents**: A service with 20 direct consumers has higher blast radius than one with 2.
- **Criticality of dependents**: If the affected services include payment processing or authentication, the business impact is severe even if the count is low.
- **Cascading failure paths**: Service A depends on B, B depends on C. If C fails, both B and A may fail -- a cascade. The length and branching factor of cascade paths determine blast radius.
- **Mitigation mechanisms**: Circuit breakers, retries with backoff, fallback logic, and redundant replicas all reduce effective blast radius. A service with a circuit breaker that opens after 5 failures and serves cached data has much lower blast radius than one that propagates errors directly.
- **Replication factor**: A service with 3 replicas can lose one without impact; a single-replica service has full blast radius on failure.

Blast radius can be estimated quantitatively using **graph-based failure simulation**: model the system as a directed graph, inject a failure at one node, and propagate failures along edges using probabilistic rules.

Sources: [NashTech: Understanding Blast Radius in Chaos Testing](https://blog.nashtechglobal.com/understanding-blast-radius-in-chaos-testing-limiting-and-measuring-impact/), [O'Reilly: Minimize Blast Radius (Chaos Engineering book)](https://www.oreilly.com/library/view/chaos-engineering/9781491988459/ch07.html)

### Graph-Based Failure Modeling

Represent the microservice architecture as a **directed acyclic graph** (DAG), where:
- **Nodes** = services (with attributes: tier, replicas, SLA criticality, circuit breaker presence)
- **Edges** = dependency relationships (with attributes: call frequency in requests/sec, failure propagation probability)

To estimate blast radius for a given node failure, run a **Monte Carlo simulation**:

1. Mark the target node as failed.
2. For each neighbor of a failed node, determine if it also fails: sample from Bernoulli(p) where p = edge's `failure_propagation_probability`, modified by mitigation factors (e.g., multiply by 0.1 if the neighbor has a circuit breaker).
3. Repeat propagation in BFS/DFS order until no new nodes fail in a round.
4. Record the total number of failed nodes.
5. Repeat steps 1-4 for N simulations (e.g., 1000) to get a distribution of blast radius.

The result is a distribution per node: mean affected nodes, standard deviation, P95, max, and which critical services appear in the cascade. This gives a much richer picture than a single deterministic estimate.

Sources: [NetworkX Centrality Documentation](https://networkx.org/documentation/stable/reference/algorithms/centrality.html), [Practical Graph Theory using NetworkX](https://garba.org/posts/2022/graph/)

### ML for Chaos Experiment Prioritization

Once we have blast radius measurements (from simulation or historical incident data), we can train a **machine learning model** to predict blast radius from graph features alone -- without running the expensive Monte Carlo simulation every time the graph changes.

**Feature extraction per node:**

| Feature | What It Captures |
|---|---|
| `in_degree` | How many services call this node |
| `out_degree` | How many services this node depends on |
| `betweenness_centrality` | How often this node lies on shortest paths between other nodes -- measures "bridge" importance |
| `pagerank` | Recursive importance: a node is important if important nodes depend on it |
| `closeness_centrality` | How quickly failures from this node can reach all others |
| `num_downstream_services` | Total reachable nodes in the dependency subgraph (direct + transitive) |
| `has_circuit_breaker` | Whether failure propagation is mitigated |
| `replicas` | Redundancy level |
| `sla_criticality` | Business importance (1-5 scale) |
| `avg_edge_propagation_prob` | Mean failure propagation probability of incoming edges |
| `tier_encoded` | Position in the architecture (frontend=0, gateway=1, backend=2, data=3, external=4) |

A **RandomForestRegressor** works well here because: (1) it handles mixed feature types naturally, (2) it provides feature importance rankings that explain *why* certain nodes are high-risk, (3) the ensemble of trees gives prediction variance, which we use as an uncertainty measure.

The model is trained on (features, blast_radius) pairs from the Monte Carlo simulation, then used to predict blast radius for new or modified nodes without re-running expensive simulations.

### Experiment Selection Under Uncertainty

Given predicted blast radius for every node, how do we choose which experiments to run? We want experiments that **maximize learning**, not just maximize damage.

The key insight is the **exploration vs. exploitation** trade-off (familiar from multi-armed bandits and reinforcement learning):
- **Exploitation**: Run experiments on nodes with highest predicted blast radius. We are confident these are impactful.
- **Exploration**: Run experiments on nodes where the model is **uncertain** about the blast radius. We don't know what will happen, so the learning value is high.

The scoring formula balances both:

```
score = predicted_impact x (1 + uncertainty_bonus) x criticality_weight
```

Where:
- `predicted_impact` = model's predicted mean blast radius for the node
- `uncertainty_bonus` = standard deviation across trees in the RandomForest (high variance = high uncertainty = high exploration value)
- `criticality_weight` = multiplier based on service's SLA criticality (failing a critical service teaches more than failing a non-critical one)

This is analogous to the **Upper Confidence Bound (UCB)** strategy in bandits: prefer actions with high estimated reward OR high uncertainty. Over time, as experiments reduce uncertainty about node blast radii, the model naturally shifts toward exploiting known high-impact targets.

### Steady State Hypothesis: Detecting Normal Behavior

A robust chaos experiment requires a well-defined **steady state hypothesis**: a quantitative definition of "normal" that can be checked before, during, and after failure injection.

The approach:
1. **Collect baseline metrics** over a window of normal operation: request latency (P50, P95, P99), error rate, throughput (requests/sec), CPU utilization, memory usage.
2. **Define bounds** using statistical methods: for each metric, compute mean and standard deviation over the baseline window. Steady state = metric stays within [mean - k*std, mean + k*std] where k is a sensitivity parameter (typically 2-3).
3. **Use rolling windows** to detect transient violations vs. sustained drift. A single spike outside bounds is not a steady-state violation; sustained deviation over a window is.
4. **Check before injection** to confirm the system is in steady state (don't inject chaos into an already-degraded system).
5. **Check during and after injection** to determine if the hypothesis holds.

The choice between **global statistics** (mean/std over the entire baseline) and **rolling statistics** (mean/std over a sliding window) matters: global statistics are stable but miss time-of-day patterns; rolling statistics adapt to recent behavior but are noisier. For most systems, rolling statistics over a window of 20-50 data points work well.

### Real-World Chaos Engineering Tools

The ecosystem has matured significantly since Chaos Monkey:

| Tool | Type | Key Strength |
|---|---|---|
| **Chaos Monkey** (Netflix) | Open source | The original; random instance termination on AWS |
| **Gremlin** | Commercial SaaS | First commercial platform; turn-key experiments with fine-grained blast radius control |
| **LitmusChaos** | Open source (CNCF) | Kubernetes-native; ChaosHub library of fault templates; Litmus Probes for health monitoring |
| **AWS Fault Injection Simulator (FIS)** | Cloud service | Deep AWS integration; controls EC2, ECS, EKS, RDS, and networking faults |
| **Steadybit** | Commercial (open extensions) | Resilience policies; auto-discovery of system topology; integrated steady-state checks |
| **Chaos Mesh** | Open source (CNCF) | Kubernetes-native; rich fault types (network, I/O, time, JVM, kernel); visual dashboard |

Sources: [Gremlin: Chaos Engineering Tools Comparison](https://www.gremlin.com/community/tutorials/chaos-engineering-tools-comparison), [Steadybit: Top Chaos Engineering Tools 2025 Guide](https://steadybit.com/blog/top-chaos-engineering-tools-worth-knowing-about-2025-guide/), [awesome-chaos-engineering](https://github.com/dastergon/awesome-chaos-engineering)

### Why ML + Chaos Engineering Matters

In large-scale systems (hundreds of microservices, evolving daily), manual chaos planning does not scale. Engineers cannot keep a mental model of all dependency chains and failure modes. ML-guided chaos engineering offers:

1. **Efficient experiment selection**: Focus on high-impact, high-uncertainty targets instead of random guessing.
2. **Quantitative risk assessment**: Feature importances reveal *why* certain nodes are critical (e.g., "high betweenness centrality" = bridge node, "low replicas + high in-degree" = single point of failure).
3. **Continuous adaptation**: As the system architecture changes (new services, new dependencies), re-extract graph features and re-predict blast radii without re-running all simulations.
4. **Proactive vulnerability detection**: The model can flag new services that exhibit high-risk feature patterns before any incident occurs.

This practice implements the core pipeline: graph modeling -> Monte Carlo simulation -> ML prediction -> intelligent experiment ranking -> steady state verification.

## Description

Build an **ML-guided Chaos Engineering pipeline** that models a microservice system as a dependency graph, simulates cascading failures via Monte Carlo methods, trains a RandomForest to predict blast radius from graph features, and ranks chaos experiments by a combined impact-uncertainty score. The system also defines steady-state hypotheses from baseline metrics and checks whether injected failures violate steady state.

### What you'll build

1. **Synthetic microservice graph** (15-20 nodes across 5 tiers) with realistic dependency structure, circuit breakers, and SLA metadata
2. **Monte Carlo cascading failure simulator** that propagates failures probabilistically along edges
3. **Blast radius prediction model** using graph centrality features and RandomForest regression
4. **Experiment prioritizer** that ranks chaos experiments by predicted impact and model uncertainty
5. **Steady state detector** that defines and checks metric baselines for normal vs. degraded operation

### What you'll learn

- How to model distributed systems as graphs and extract structural features
- How Monte Carlo simulation quantifies failure impact under uncertainty
- How ML can prioritize chaos experiments for maximum learning
- How to define and check steady-state hypotheses quantitatively
- The exploration vs. exploitation trade-off in experiment selection

## Instructions

### Exercise 1: Monte Carlo Cascading Failure Simulation (~25 min)

**File:** `src/01_failure_simulation.py`

This exercise teaches how cascading failures propagate in a dependency graph. When a service fails, its dependents may also fail depending on the strength of the coupling, the presence of circuit breakers, and randomness. Monte Carlo simulation (running many random trials) converts this stochastic process into a distribution of outcomes -- giving you not just "how many services fail on average" but the full spread of possible outcomes.

**What to implement:**
- `simulate_single_failure()`: BFS-based failure propagation with probabilistic edge failure and circuit breaker mitigation
- `compute_blast_radius()`: Aggregate simulation results into statistical summary metrics (mean, std, max, P95, critical count)

**Why it matters:** This is how chaos engineering tools estimate the "blast radius" of an experiment before running it. Understanding this simulation is key to understanding why some services are more dangerous to fail than others.

### Exercise 2: Blast Radius Prediction Model (~25 min)

**File:** `src/02_blast_radius_model.py`

This exercise teaches feature engineering on graphs and supervised learning. Instead of running expensive Monte Carlo simulations every time the system changes, we extract structural features (centrality measures, topology metrics) and train a model to predict blast radius directly from these features.

**What to implement:**
- `extract_node_features()`: Compute graph centrality measures (betweenness, PageRank, closeness) and service metadata features
- `train_blast_radius_model()`: Train a RandomForestRegressor with cross-validation and feature importance analysis

**Why it matters:** Feature engineering on graphs is a broadly applicable skill. The model's feature importances reveal *why* certain nodes are critical -- enabling architects to make informed decisions about where to add redundancy.

### Exercise 3: Experiment Prioritizer (~15 min)

**File:** `src/03_experiment_prioritizer.py`

This exercise teaches the exploration-exploitation trade-off in the context of experiment selection. High-impact experiments are valuable, but so are experiments where the model is uncertain -- because that is where the most learning happens.

**What to implement:**
- `score_experiment()`: Combine predicted impact, model uncertainty, and criticality into a priority score
- `rank_experiments()`: Generate a ranked list of chaos experiments using the trained model

**Why it matters:** This is the core insight of ML-guided chaos: instead of random experiments, focus on where you will learn the most. The same principle applies to active learning, A/B testing, and any domain where experiments are expensive.

### Exercise 4: Steady State Detection (~15 min)

**File:** `src/04_steady_state.py`

This exercise teaches how to define "normal" quantitatively and check whether a system deviates from it. This is the foundational capability that makes chaos experiments meaningful -- without a steady state definition, you cannot tell whether an experiment succeeded or failed.

**What to implement:**
- `define_steady_state()`: Compute statistical bounds (mean +/- k*std) from baseline metrics using rolling windows
- `check_steady_state()`: Verify that metrics stay within bounds over a given time window

**Why it matters:** Steady state detection is not just for chaos engineering -- it is the basis of anomaly detection, SLO monitoring, and automated incident detection. The rolling-window vs. global-statistics design choice appears in many monitoring systems.

## Motivation

- **SRE/Platform engineering skill**: Chaos engineering is a core practice in Site Reliability Engineering. Understanding how to systematically identify and prioritize failure scenarios is essential for building resilient distributed systems.
- **Graph ML + systems thinking**: Extracting features from dependency graphs and using them for prediction bridges graph theory, ML, and distributed systems -- three domains that increasingly overlap in production systems.
- **Complements practices 014 (SAGA) and 052 (Resilience Patterns)**: Those practices build resilient patterns; this one tests whether those patterns actually work under failure.
- **Industry demand**: Companies like Netflix, Google, Amazon, and Uber all practice chaos engineering. The trend toward ML-guided chaos (Gremlin's intelligent recommendations, Steadybit's auto-discovery) makes this intersection increasingly relevant.

## Commands

| Command | Description |
|---------|-------------|
| `uv sync` | Install all dependencies (networkx, scikit-learn, numpy, pandas, matplotlib) |
| `uv run python src/00_system_model.py` | Build synthetic microservice graph, generate baseline metrics, save to data/ |
| `uv run python src/01_failure_simulation.py` | Run Monte Carlo failure simulation for every node, save blast radii to data/ |
| `uv run python src/02_blast_radius_model.py` | Extract graph features, train RandomForest blast radius model, save to models/ |
| `uv run python src/03_experiment_prioritizer.py` | Rank chaos experiments by impact x uncertainty, display prioritized list |
| `uv run python src/04_steady_state.py` | Define steady state bounds, inject simulated chaos, check for violations |

## State

`not-started`
