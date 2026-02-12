# System Design Trade-Offs — Detailed Reference

For each trade-off, this guide explains **both sides**, **when to choose each**, and **real-world examples**.

---

## 1. Consistency vs Availability (CAP Theorem)

**The rule:** In a network partition, you must choose between Consistency (all nodes see the same data) or Availability (every request gets a response). You can't have both during a partition.

| Choose Consistency (CP) | Choose Availability (AP) |
|------------------------|------------------------|
| Financial transactions | Social media feeds |
| Inventory (prevent overselling) | DNS resolution |
| Leader election | Shopping cart (merge later) |
| Distributed locks | Analytics dashboards |

**In practice:** Most systems are not purely CP or AP — they tune consistency per operation. Example: Amazon's shopping cart is AP (always writable), but checkout is CP (verify inventory).

**PACELC extension:** Even without partitions, there's a trade-off between Latency and Consistency. Low-latency reads often require eventual consistency (read from nearest replica).

---

## 2. Latency vs Throughput

**The tension:** Batching increases throughput but adds latency. Immediate processing minimizes latency but limits throughput.

| Optimize Latency | Optimize Throughput |
|-----------------|-------------------|
| Real-time chat, gaming | Batch ETL pipelines |
| HFT order execution | Log aggregation |
| Autocomplete/typeahead | Report generation |
| Payment authorization | Email sending |

**Techniques for both:**
- **Batching with timeout:** Collect up to N items OR wait T ms, whichever comes first. Kafka's `linger.ms` does this.
- **Async processing:** Return immediately, process in background. Good UX latency, good backend throughput.

---

## 3. SQL vs NoSQL

| SQL (Relational) | NoSQL (Various) |
|-----------------|----------------|
| Complex joins, aggregations | Simple key-value or document lookups |
| ACID transactions needed | Eventual consistency acceptable |
| Schema is well-defined, stable | Schema evolves rapidly |
| Vertical scaling sufficient | Horizontal scaling required |
| Moderate write volume | Very high write volume |

**NoSQL subtypes:**
- **Key-Value** (Redis, DynamoDB): Fastest lookups, no query flexibility
- **Document** (MongoDB): Flexible schema, nested data, no joins
- **Wide-Column** (Cassandra, HBase): Time-series, write-heavy, column families
- **Graph** (Neo4j): Relationship-heavy queries (social graphs, recommendations)

**Common mistake:** Choosing NoSQL "for scale" when PostgreSQL with read replicas handles 90% of use cases.

---

## 4. Synchronous vs Asynchronous Communication

| Synchronous (Request-Response) | Asynchronous (Event/Message) |
|-------------------------------|----------------------------|
| Simple, easy to reason about | Decoupled, resilient to failures |
| Immediate consistency | Eventual consistency |
| Cascading failures possible | Natural backpressure |
| Tight coupling between services | Loose coupling |

**Hybrid pattern:** Synchronous for reads (user expects immediate response), asynchronous for writes (publish event, process later). Example: POST /order returns 202 Accepted, order processing happens via message queue.

---

## 5. Push vs Pull

| Push (Server-Initiated) | Pull (Client-Initiated) |
|------------------------|----------------------|
| Low latency for updates | Client controls pace |
| Server manages connections | Simpler server, stateless |
| WebSocket, SSE, webhooks | REST polling, batch pull |
| Notification systems | Feed generation, batch jobs |

**Fan-out problem:** When a user with 10M followers posts, push means 10M writes. Pull means computing the feed on read. Solution: **hybrid fan-out** — push to active users, pull for inactive.

---

## 6. Monolith vs Microservices

| Monolith | Microservices |
|----------|--------------|
| Simple deployment, one binary | Independent deployment per service |
| Easy debugging (single process) | Service-level scaling |
| Fast inter-module calls | Network overhead, partial failures |
| Small team (< 10 devs) | Large org, many teams |
| Early-stage product | Mature product with clear boundaries |

**Conway's Law:** System architecture mirrors team structure. Microservices work when team boundaries align with service boundaries.

**Common mistake:** Starting with microservices. Start monolith, extract services when you identify clear boundaries and scaling bottlenecks.

---

## 7. Normalization vs Denormalization

| Normalized | Denormalized |
|-----------|-------------|
| No data duplication | Precomputed joins, duplicated data |
| Write-efficient (update one place) | Read-efficient (no joins at query time) |
| Complex reads (joins) | Complex writes (update multiple copies) |
| OLTP workloads | Read-heavy, latency-sensitive |

**Denormalization strategies:**
- **Materialized views:** Precomputed query results, refreshed periodically or on write.
- **CQRS:** Separate read/write models. Write model is normalized, read model is denormalized.
- **Embedding:** Store related data together (e.g., user + recent orders in one document).

---

## 8. Replication vs Sharding

| Replication (Copies) | Sharding (Splits) |
|---------------------|------------------|
| Same data on multiple nodes | Different data on different nodes |
| Read scaling | Write scaling |
| High availability | Larger dataset capacity |
| Consistency challenges | Cross-shard query challenges |

**Typically combined:** Shard the data for write scale, replicate each shard for read scale and availability.

---

## 9. Precomputation vs On-Demand

| Precompute | On-Demand |
|-----------|-----------|
| Predictable latency | Flexible, handles any query |
| Wasted compute for unused results | Cold-start latency |
| News feeds, dashboards, leaderboards | Search, ad-hoc analytics |
| Stale data until refresh | Always fresh |

**Hybrid:** Precompute common queries, fall back to on-demand for long-tail.

---

## 10. Build vs Buy

| Build | Buy/Use Managed |
|-------|----------------|
| Full control, custom optimization | Faster time-to-market |
| Ongoing maintenance burden | Vendor lock-in |
| HFT (where μs matter) | Most SaaS companies |
| Core competitive advantage | Commodity infrastructure |

**Staff+ signal:** Knowing when NOT to build. "We should use managed Kafka (Confluent) because maintaining Kafka clusters is not our core competency" is a stronger answer than "let me design a custom message queue."
