# Practice 027: Advanced System Design Interview

## Technologies

- **System Design** — distributed systems architecture, scalability, reliability patterns
- **RESHADED Framework** — 8-step structured approach to system design interviews
- **DDIA Concepts** — Designing Data-Intensive Applications core patterns

## Stack

- Markdown (design documents, no code)
- Pen & paper / whiteboard encouraged

## Theoretical Context

### What Is a System Design Interview?

A system design interview tests your ability to architect large-scale distributed systems under ambiguous requirements. Unlike coding interviews (which test algorithms), design interviews evaluate **trade-off reasoning**, **communication**, and **breadth of systems knowledge**. You're given an open-ended prompt ("Design Twitter") and 45-60 minutes to collaboratively build an architecture with your interviewer.

### The RESHADED Framework

A structured 8-step approach to ensure comprehensive coverage:

| Step | Name | What to Do |
|------|------|------------|
| **R** | Requirements | Clarify functional & non-functional requirements. Ask questions. Scope to 2-3 core features. |
| **E** | Estimation | Back-of-the-envelope: QPS, storage, bandwidth. Identify scale tier (thousands → millions → billions). |
| **S** | Storage Schema | Data model: entities, relationships, access patterns. SQL vs NoSQL decision. |
| **H** | High-Level Design | Box diagram: clients, services, databases, caches, queues. Show data flow. |
| **A** | APIs | Key endpoint signatures. REST/gRPC/GraphQL choice. Request/response shapes. |
| **D** | Detailed Design | Deep dive into 2-3 hardest components. This is where you differentiate. |
| **E** | Evaluation | Trade-offs, failure modes, bottlenecks. What would break first at 10x scale? |
| **D** | Distinctive Component | The unique/novel part of your design. Shows creativity and depth. |

**Source:** [Educative — RESHADED Approach](https://www.educative.io/blog/use-reshaded-for-system-design-interviews)

### Level Expectations

| Level | Focus | Time Horizon | Key Signal |
|-------|-------|-------------|------------|
| **Mid** | Functional requirements, basic architecture, known patterns | Current needs | Can build with guidance |
| **Senior** | Trade-offs, scale, failure modes, back-of-envelope math | 6-12 months | Can design independently |
| **Staff+** | Simplicity over complexity, organizational impact, build vs buy | 1-2 years | Can lead architecture decisions |

**Source:** [Hello Interview — What Is Expected at Each Level](https://www.hellointerview.com/blog/the-system-design-interview-what-is-expected-at-each-level)

### Core Trade-Offs

These trade-offs appear in virtually every system design discussion:

| Trade-Off | When to Choose A | When to Choose B |
|-----------|-----------------|-----------------|
| **Consistency vs Availability** (CAP) | Financial transactions, inventory | Social feeds, analytics |
| **Latency vs Throughput** | Real-time chat, HFT | Batch ETL, report generation |
| **Push vs Pull** | Low-latency events, serverless | Batch consumers, backpressure control |
| **SQL vs NoSQL** | Complex queries, ACID needs | Flexible schema, horizontal scale |
| **Sync vs Async** | Immediate consistency | Decoupled services, resilience |
| **Monolith vs Microservices** | Small team, early stage | Independent scaling, large teams |
| **Cache vs Source-of-Truth** | Read-heavy, latency-sensitive | Write-heavy, consistency-critical |

### Common Mistakes

1. **Jumping into design** without clarifying requirements
2. **Poor scoping** — trying to design all of Twitter in 45 minutes
3. **Ignoring trade-offs** — saying "use Kafka" without explaining why
4. **Staying silent** — not communicating your thought process
5. **Memorizing architectures** — pattern-matching instead of reasoning from first principles
6. **Over-focusing on low-level details** — class definitions instead of architecture

**Sources:** [Educative — 6 Common Mistakes](https://www.educative.io/blog/six-common-system-design-interview-mistakes), [DesignGurus — Common Mistakes](https://www.designgurus.io/answers/detail/what-are-the-common-mistakes-in-a-system-design-interview)

## Description

A curated workbook of **20 advanced system design problems** organized into 4 tiers by complexity. Each problem includes what it tests, scale constraints, key questions to address, and a structured TODO(human) section for writing your design.

This is a **thinking practice** — no code. For each problem, work through the RESHADED framework and write your design decisions, trade-offs, and architecture in the exercise files.

### What you'll practice

1. **Structured problem decomposition** — using RESHADED to avoid missing key aspects
2. **Trade-off reasoning** — articulating why you choose A over B
3. **Back-of-the-envelope estimation** — QPS, storage, bandwidth calculations
4. **Deep-dive selection** — identifying and diving into the 2-3 hardest components
5. **Failure mode analysis** — what breaks, how to recover, graceful degradation
6. **Communication** — explaining architecture decisions clearly and concisely

### Exercise Tiers

| Tier | Focus | Exercises |
|------|-------|-----------|
| 1 — Foundational | Core patterns every engineer must know | 5 problems |
| 2 — Infrastructure | Internal platform and infrastructure systems | 5 problems |
| 3 — Complex | Multi-domain systems with competing requirements | 5 problems |
| 4 — Specialized | Modern/niche: HFT, ML infra, blockchain-adjacent | 5 problems |

## Instructions

### How to Use This Practice

This is **not** a single 60-120 min session. It's a workbook you return to over multiple sessions.

**Per problem (~30-45 min):**
1. Read the problem statement and constraints
2. Work through RESHADED (optionally on paper first)
3. Write your design in the TODO(human) section of the exercise file
4. Review the "Key Questions" — can you answer all of them?
5. Discuss with Claude to verify your reasoning and explore alternatives

**Suggested order:**
- Start with Tier 1 (foundational patterns)
- Move to Tier 2-3 based on target company type
- Tier 4 for specialized roles (HFT, ML infra)

### Reference Material

Before starting exercises, read through the reference files:
- `reference/concepts.md` — Quick reference for distributed systems concepts (consistency models, consensus, sharding, caching, etc.)
- `reference/trade_offs.md` — Detailed trade-off analysis for common decisions

### Exercise Format

Each exercise follows this structure:
```
## Problem: Design X
**Tests:** [key concepts]
**Scale:** [expected scale constraints]

### Key Questions
- [specific questions to address in your design]

### TODO(human): Your Design
<!-- Write your design here following RESHADED -->
```

## Motivation

- **Interview readiness**: System design is the highest-weight interview round for senior+ roles at FAANG, HFT, and infrastructure companies
- **Architectural thinking**: Forces structured reasoning about trade-offs, failure modes, and scale — directly applicable to production work
- **Knowledge gaps**: Identifies which distributed systems concepts need deeper study
- **Complementary to coding practices**: Practices 002-026 build implementation skills; this practice builds architectural reasoning

## Commands

Since this is a thinking/writing practice, there are no build/run commands.

| Command | Description |
|---------|-------------|
| Open any `exercises/tier*.md` file | Read problem statements and write your designs |
| Open `reference/concepts.md` | Quick lookup for distributed systems concepts |
| Open `reference/trade_offs.md` | Detailed trade-off analysis reference |

## References

### Books
- [Designing Data-Intensive Applications (Martin Kleppmann)](https://dataintensive.net/) — The definitive distributed systems reference
- [System Design Interview Vol. 1 & 2 (Alex Xu)](https://www.amazon.com/System-Design-Interview-insiders-Second/dp/B08CMF2CQF) — 16 problems per volume, visual, practical

### Courses & Guides
- [Grokking the System Design Interview (Educative)](https://www.educative.io/courses/grokking-the-system-design-interview) — Pattern-based, 140K+ learners
- [Hello Interview — System Design](https://www.hellointerview.com/learn/system-design) — Level-specific expectations
- [A Senior Engineer's Guide to System Design (interviewing.io)](https://interviewing.io/guides/system-design-interview) — Communication and leadership signals

### Concept Deep-Dives
- [Linearizability vs Serializability (SystemDesign School)](https://systemdesignschool.io/blog/linearizability-vs-serializability)
- [Paxos vs Raft vs ZAB (Medium)](https://medium.com/@remisharoon/paxos-vs-raft-vs-zab-a-comprehensive-dive-into-distributed-consensus-protocols-6243a3f6539b)
- [The Big Little Guide to Message Queues](https://sudhir.io/the-big-little-guide-to-message-queues)
- [CAP Theorem for System Design (Hello Interview)](https://www.hellointerview.com/learn/system-design/core-concepts/cap-theorem)

## State

`not-started`
