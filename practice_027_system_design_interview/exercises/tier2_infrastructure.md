# Tier 2 — Infrastructure System Design Problems

Internal platform and infrastructure systems. These test deeper understanding of distributed systems internals.

---

## Problem 6: Design a Distributed Task Scheduler (cron at scale)

**Tests:** Distributed coordination, exactly-once execution, fault tolerance, priority queues
**Scale:** 10M scheduled tasks, 100K executions/min, tasks must execute within 1s of scheduled time

### Key Questions
- How do you ensure a task runs exactly once even with multiple scheduler instances?
- How do you handle scheduler node failure mid-execution?
- Priority scheduling — how do you prevent starvation of low-priority tasks?
- How do you handle a burst of tasks all scheduled for the same time?
- Recurring tasks (cron expressions) vs one-shot tasks — different storage/execution paths?
- How does a user cancel or modify a scheduled task?

### TODO(human): Your Design

<!-- Requirements (functional + non-functional): -->


<!-- Task storage + indexing strategy: -->


<!-- Distributed locking / leader election for execution: -->


<!-- High-level design: -->


<!-- Exactly-once execution guarantee: -->


<!-- Failure recovery (scheduler crash, worker crash): -->


<!-- Scaling strategy (10M → 1B tasks): -->


---

## Problem 7: Design a Search Autocomplete / Typeahead System

**Tests:** Trie/inverted index, ranking, caching, latency optimization, data freshness
**Scale:** 10B queries/day, < 50ms p99 latency, suggestions update within minutes of trending

### Key Questions
- Trie vs inverted index with n-grams — which and why?
- How do you rank suggestions? (frequency, personalization, recency)
- How do you handle trending queries in near-real-time?
- Caching strategy — at which layer? Browser, CDN, application?
- Multi-language support — how does it affect the data structure?
- How do you filter offensive/inappropriate suggestions?

### TODO(human): Your Design

<!-- Requirements (functional + non-functional): -->


<!-- Data structure choice + justification: -->


<!-- Ranking algorithm: -->


<!-- High-level design: -->


<!-- Data pipeline (how queries become suggestions): -->


<!-- Caching layers: -->


<!-- Freshness vs latency trade-off: -->


---

## Problem 8: Design a Notification System

**Tests:** Multi-channel delivery, priority, deduplication, rate limiting per user, template management
**Scale:** 1B notifications/day across push, SMS, email; < 5s delivery for high-priority

### Key Questions
- How do you route notifications to the right channel (push, SMS, email)?
- User preferences (opt-in/opt-out, quiet hours, frequency caps) — where stored, how enforced?
- Priority levels — how do you ensure critical alerts aren't delayed by marketing notifications?
- Deduplication — same event triggers from multiple services, user sees only one notification?
- How do you handle delivery failures and retries per channel?
- Template rendering — where does it happen and how do you support A/B testing?

### TODO(human): Your Design

<!-- Requirements (functional + non-functional): -->


<!-- High-level design (ingestion → routing → delivery): -->


<!-- Channel routing logic: -->


<!-- Priority queue design: -->


<!-- Deduplication strategy: -->


<!-- Failure handling per channel: -->


<!-- User preference enforcement: -->


---

## Problem 9: Design a CDN (Content Delivery Network)

**Tests:** Edge caching, cache invalidation, DNS routing, origin shielding, consistency
**Scale:** Global distribution, 100K+ edge nodes, 1M requests/sec, < 50ms TTFB

### Key Questions
- How does a client request get routed to the nearest edge node? (DNS-based vs Anycast)
- Cache hierarchy: edge → regional → origin shield → origin. Why multiple layers?
- Cache invalidation: TTL vs purge API vs versioned URLs — trade-offs?
- Cache key design — what do you include? (URL, headers, cookies, query params)
- How do you handle cache stampede on a popular object expiring?
- Dynamic content (personalized pages) — can you still CDN-cache them?

### TODO(human): Your Design

<!-- Requirements (functional + non-functional): -->


<!-- Routing strategy (how requests reach the right edge): -->


<!-- Cache hierarchy design: -->


<!-- High-level design: -->


<!-- Invalidation strategy: -->


<!-- Cache stampede mitigation: -->


<!-- Dynamic vs static content handling: -->


---

## Problem 10: Design a Monitoring & Alerting System (Datadog/Prometheus)

**Tests:** Time-series data, aggregation at scale, anomaly detection, alert routing
**Scale:** 10M metrics/sec ingestion, 1-year retention, query latency < 1s for dashboards

### Key Questions
- Time-series storage: dedicated TSDB vs general-purpose DB? Why?
- How do you handle 10M metrics/sec ingestion without losing data?
- Downsampling: keep 1s resolution for 24h, 1min for 30 days, 1h for 1 year — how?
- Alert evaluation: pull-based (Prometheus) vs push-based — trade-offs?
- How do you prevent alert storms? (grouping, inhibition, silencing)
- Multi-tenant: how do you isolate customers' metrics while sharing infrastructure?

### TODO(human): Your Design

<!-- Requirements (functional + non-functional): -->


<!-- Ingestion pipeline: -->


<!-- Storage engine (TSDB design): -->


<!-- High-level design: -->


<!-- Downsampling strategy: -->


<!-- Alert evaluation engine: -->


<!-- Query engine for dashboards: -->
