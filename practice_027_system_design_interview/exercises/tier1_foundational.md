# Tier 1 — Foundational System Design Problems

These are the most commonly asked problems. Every senior engineer should be able to design these fluently.

---

## Problem 1: Design a URL Shortener (bit.ly)

**Tests:** Hashing, distributed ID generation, caching, read-heavy scaling, analytics
**Scale:** 100M URLs created/month, 10B redirects/month, 10:1 read:write ratio

### Key Questions
- How do you generate short, unique URLs? (hash vs counter vs Snowflake)
- How do you handle hash collisions?
- How do you scale reads to handle 10B redirects/month (~3800 QPS)?
- How would you add analytics (click count, geo, referrer) without slowing redirects?
- What's your data model and why? (SQL vs KV store)
- Custom aliases — how do you prevent conflicts?

### TODO(human): Your Design

<!-- Requirements (functional + non-functional): -->


<!-- Estimation (QPS, storage, bandwidth): -->


<!-- Storage schema: -->


<!-- High-level design (components + data flow): -->


<!-- API design: -->


<!-- Deep dive (pick 2-3 hard parts): -->


<!-- Trade-offs and failure modes: -->


---

## Problem 2: Design a Distributed Rate Limiter

**Tests:** Rate limiting algorithms, distributed coordination, low-latency decision making, consistency vs performance
**Scale:** 1M+ API clients, < 1ms decision latency, multi-datacenter

### Key Questions
- Which algorithm? (token bucket vs sliding window — why?)
- How do you distribute rate limit state across multiple API gateway instances?
- Redis-based vs local-with-sync — what are the trade-offs?
- How do you handle clock skew across instances?
- What happens when the rate limiter itself fails? (fail-open vs fail-closed)
- How do you support different rate limits per tier (free/paid/enterprise)?

### TODO(human): Your Design

<!-- Requirements (functional + non-functional): -->


<!-- Estimation (QPS, storage): -->


<!-- Algorithm choice + justification: -->


<!-- High-level design: -->


<!-- Deep dive — distributed state: -->


<!-- Failure modes (rate limiter down, Redis down, split-brain): -->


---

## Problem 3: Design a Distributed Key-Value Store

**Tests:** Partitioning, replication, consistency models, failure detection, read/write paths
**Scale:** Billions of keys, sub-10ms p99 latency, 99.99% availability

### Key Questions
- Consistent hashing for partitioning — how do you handle hotspots and rebalancing?
- Replication factor N=3, quorum reads/writes (W + R > N) — why?
- How do you detect and handle node failures? (gossip protocol, heartbeats)
- Read repair vs anti-entropy — when do you use each?
- How do you handle conflicting writes? (vector clocks, last-write-wins, CRDTs)
- Tunable consistency: how does the client choose strong vs eventual per request?

### TODO(human): Your Design

<!-- Requirements (functional + non-functional): -->


<!-- Data partitioning strategy: -->


<!-- Replication + consistency model: -->


<!-- High-level design (coordinator, storage nodes, failure detector): -->


<!-- Write path (step by step): -->


<!-- Read path (step by step): -->


<!-- Failure handling (node down, network partition): -->


---

## Problem 4: Design a Chat System (WhatsApp/Slack)

**Tests:** Real-time messaging, presence, message delivery guarantees, fan-out, offline handling
**Scale:** 500M DAU, 50B messages/day, < 100ms delivery latency

### Key Questions
- WebSocket vs SSE vs long polling for real-time delivery?
- How do you guarantee message delivery (at-least-once) for offline users?
- Message ordering — per-conversation or global? How do you handle out-of-order?
- Group chat fan-out: small groups (< 100) vs large channels (> 10K)?
- How do you implement "last seen" / typing indicators without overwhelming the system?
- End-to-end encryption — where does the server fit?
- How do you store and retrieve chat history efficiently?

### TODO(human): Your Design

<!-- Requirements (functional + non-functional): -->


<!-- Estimation (messages/sec, connections, storage): -->


<!-- High-level design: -->


<!-- Real-time delivery mechanism: -->


<!-- Message storage + retrieval: -->


<!-- Offline message handling: -->


<!-- Group chat fan-out strategy: -->


<!-- Presence system design: -->


---

## Problem 5: Design a News Feed System (Twitter/Facebook)

**Tests:** Fan-out strategies, ranking, caching, read vs write optimization, celebrity problem
**Scale:** 500M DAU, average user follows 200 accounts, feed refreshed every 30s

### Key Questions
- Fan-out on write vs fan-out on read — when do you use each?
- The celebrity problem: a user with 50M followers posts — how do you handle it?
- How do you rank feed items? (chronological vs engagement-based)
- Feed caching strategy — what's cached, where, and for how long?
- How do you handle new follows/unfollows in real-time?
- Pagination: cursor-based vs offset — why?

### TODO(human): Your Design

<!-- Requirements (functional + non-functional): -->


<!-- Estimation (QPS for feed reads, fan-out writes): -->


<!-- Fan-out strategy + celebrity handling: -->


<!-- High-level design: -->


<!-- Feed storage + caching: -->


<!-- Ranking approach: -->


<!-- Trade-offs (latency vs freshness vs cost): -->
