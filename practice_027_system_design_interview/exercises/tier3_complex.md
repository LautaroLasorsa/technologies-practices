# Tier 3 — Complex System Design Problems

Multi-domain systems with competing requirements. These test your ability to manage complexity and make principled trade-offs across multiple dimensions.

---

## Problem 11: Design a Ride-Sharing Service (Uber/Lyft)

**Tests:** Geospatial indexing, real-time matching, ETA calculation, surge pricing, consistency
**Scale:** 10M rides/day, 1M concurrent drivers, matching within 10s, ETA accuracy within 2 min

### Key Questions
- How do you efficiently find nearby drivers? (geohash, quadtree, R-tree)
- Matching algorithm: nearest driver vs optimal assignment (minimize global wait time)?
- How do you calculate and update ETAs in real-time with traffic?
- Surge pricing: how do you detect high demand zones and adjust prices?
- Ride state machine: how do you ensure consistency (no double-assign, no lost rides)?
- How do you handle driver location updates at 1M drivers reporting every 3 seconds?

### TODO(human): Your Design

<!-- Requirements (functional + non-functional): -->


<!-- Geospatial indexing strategy: -->


<!-- Matching system design: -->


<!-- High-level design: -->


<!-- Location tracking at scale: -->


<!-- ETA calculation: -->


<!-- Surge pricing mechanism: -->


---

## Problem 12: Design Google Docs (Collaborative Editor)

**Tests:** Real-time collaboration, conflict resolution (OT/CRDT), presence, versioning
**Scale:** 100M documents, 10K concurrent editors on popular docs, < 100ms sync latency

### Key Questions
- OT (Operational Transform) vs CRDT — which and why?
- How do you handle concurrent edits to the same paragraph by multiple users?
- Cursor/selection presence — how do you show where other users are editing?
- Offline editing — how do you sync changes when a user reconnects?
- Version history — how do you store and retrieve any past version efficiently?
- Permission model — view, comment, edit at document and section level?

### TODO(human): Your Design

<!-- Requirements (functional + non-functional): -->


<!-- Conflict resolution strategy (OT vs CRDT): -->


<!-- Real-time sync protocol: -->


<!-- High-level design: -->


<!-- Document storage + versioning: -->


<!-- Presence system: -->


<!-- Offline support + reconciliation: -->


---

## Problem 13: Design a Payment System (Stripe-like)

**Tests:** ACID transactions, idempotency, reconciliation, fraud detection, multi-currency
**Scale:** 1M transactions/day, 99.999% reliability, regulatory compliance (PCI-DSS)

### Key Questions
- How do you ensure a payment is processed exactly once? (idempotency keys)
- Double-entry bookkeeping — why is it essential and how do you implement it?
- Multi-step payment flow (auth → capture → settle) — how do you handle partial failures?
- How do you handle currency conversion and rounding?
- Fraud detection — real-time vs batch? What signals do you use?
- Reconciliation: how do you verify your records match the payment processor's?
- PCI-DSS compliance: what data can you store and how?

### TODO(human): Your Design

<!-- Requirements (functional + non-functional): -->


<!-- Payment flow state machine: -->


<!-- Idempotency implementation: -->


<!-- High-level design: -->


<!-- Double-entry ledger: -->


<!-- Fraud detection strategy: -->


<!-- Reconciliation process: -->


<!-- Security / PCI-DSS considerations: -->


---

## Problem 14: Design a Video Streaming Platform (YouTube/Netflix)

**Tests:** Video processing pipeline, adaptive bitrate, CDN, recommendation, cost optimization
**Scale:** 1B videos, 500M DAU, 1M video uploads/day, global delivery

### Key Questions
- Upload pipeline: how do you process a raw video into multiple resolutions/codecs?
- Adaptive bitrate streaming (HLS/DASH) — how does the client switch quality?
- CDN strategy for video: pre-warm popular content vs cache-on-demand?
- How do you handle live streaming vs VOD differently?
- Cost optimization: storage tiers (hot/warm/cold) for videos by popularity?
- Recommendation engine: what signals feed it? (watch history, engagement, social graph)

### TODO(human): Your Design

<!-- Requirements (functional + non-functional): -->


<!-- Video upload + processing pipeline: -->


<!-- Storage strategy (raw, transcoded, thumbnails): -->


<!-- High-level design: -->


<!-- Streaming delivery (CDN + adaptive bitrate): -->


<!-- Live vs VOD architecture differences: -->


<!-- Cost optimization strategy: -->


---

## Problem 15: Design a Ticket Booking System (concerts, flights)

**Tests:** Inventory consistency, reservation pattern, high contention, fairness
**Scale:** 50K concurrent users for popular events, 100K seats, zero overselling

### Key Questions
- How do you prevent overselling (two users book the last seat simultaneously)?
- Temporary holds/reservations with expiry — how do you implement and clean up?
- Queue-based fairness: FIFO vs lottery for high-demand events?
- Seat selection: how do you show real-time availability to 50K concurrent users?
- Payment timeout: user selected seats but hasn't paid in 10 min — what happens?
- Scalability: 1000 events happening simultaneously, each with different inventory patterns?

### TODO(human): Your Design

<!-- Requirements (functional + non-functional): -->


<!-- Inventory management (preventing overselling): -->


<!-- Reservation + hold mechanism: -->


<!-- High-level design: -->


<!-- High-contention handling (hot seats): -->


<!-- Fairness / queue design: -->


<!-- Payment timeout + cleanup: -->
