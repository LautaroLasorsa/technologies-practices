# Practice 080: Feature Flags & Runtime Configuration: OpenFeature & Flagsmith

## Technologies

- **OpenFeature** -- CNCF-incubating vendor-agnostic specification for feature flag evaluation
- **Flagsmith** -- Open-source feature flag and remote config service (self-hosted via Docker)
- **FastAPI** -- HTTP endpoints serving flag-controlled features
- **Docker Compose** -- Multi-service orchestration (Flagsmith + PostgreSQL + app services)

## Stack

- Python 3.12+ (uv)
- Docker / Docker Compose

## Theoretical Context

### Feature Flags: Decoupling Deployment from Release

Feature flags (also called feature toggles) are conditional statements in code that control whether a feature is active at runtime, without redeploying. They solve a fundamental problem in continuous delivery: how to deploy code to production without immediately exposing it to all users. By wrapping features in flag checks, teams can deploy dark (code present but inactive), then enable features gradually, per-user, or per-segment -- and instantly disable them if something goes wrong.

Martin Fowler's [taxonomy](https://martinfowler.com/articles/feature-toggles.html) classifies flags by longevity and dynamism:

| Category | Lifespan | Example |
|----------|----------|---------|
| **Release Toggle** | Days-weeks | Hide unfinished feature until ready |
| **Experiment Toggle** | Days-weeks | A/B test two checkout flows |
| **Ops Toggle** | Permanent-ish | Kill switch for expensive computation |
| **Permission Toggle** | Permanent | Premium features for paying users |

Internally, a flag evaluation works like this: the application asks a flag management system "is feature X enabled for this context?" The system evaluates the flag's rules (targeting rules, percentage rollout, segments) against the provided context (user ID, email, plan, country, etc.) and returns a value. That value can be a simple boolean, a string variant, a number, or a JSON object. The application then branches on this value.

### Flag Evaluation Patterns

**Boolean flags** are the simplest: on/off. Used for kill switches and basic release toggles.

**Multivariate flags** return one of several string/JSON values. Used for A/B experiments (variant "control" vs "treatment-a" vs "treatment-b") and remote configuration (change button color, API endpoint, algorithm parameters without redeployment).

**Percentage-based rollouts** expose a feature to N% of users. The flag system hashes the user identifier to deterministically assign each user to a bucket, ensuring the same user always sees the same variant (sticky bucketing). This is critical for consistent user experience and valid experiment results.

**User-targeted flags** evaluate rules against user attributes. For example: "enable for users where `plan == 'enterprise'`" or "enable for users in region `eu-west`". The flag system matches the evaluation context (user attributes) against segment definitions.

### OpenFeature: The Vendor-Agnostic Standard

[OpenFeature](https://openfeature.dev/) is a CNCF Incubating project that defines a standard API for feature flag evaluation. It solves vendor lock-in: without it, switching from LaunchDarkly to Flagsmith means rewriting every flag evaluation call. With OpenFeature, you code against a single API and swap the backend by changing the "provider."

**Architecture:**

```
Your Application
    |
    v
OpenFeature SDK (standard API)
    |
    v
Provider (Flagsmith, LaunchDarkly, flagd, etc.)
    |
    v
Flag Management System (stores flag configs, rules, segments)
```

**Key concepts:**

| Concept | Description |
|---------|-------------|
| **API** | Global entry point. Registers providers, creates clients. `api.set_provider(...)` |
| **Client** | Bound to a provider. Evaluates flags: `client.get_boolean_value("my-flag", False)` |
| **Provider** | Translates OpenFeature calls into vendor-specific API calls. One provider per backend. |
| **Evaluation Context** | User/request attributes passed to the provider for targeting. Contains `targeting_key` + custom attributes. |
| **Hooks** | Lifecycle callbacks (`before`, `after`, `error`, `finally_after`) for cross-cutting concerns (logging, telemetry, validation). |
| **Flag Evaluation Details** | Full response including value, variant, reason, and metadata. |

**Evaluation lifecycle:**

1. `before` hooks run (can modify evaluation context)
2. Provider resolves the flag value using its backend
3. `after` hooks run (can validate/log the result)
4. If error at any point: `error` hooks run
5. `finally_after` hooks always run (cleanup)

The Python SDK: `openfeature-sdk` provides `api`, `OpenFeatureClient`, `EvaluationContext`, and `Hook` base class.

### Flagsmith: Self-Hosted Feature Flag Platform

[Flagsmith](https://www.flagsmith.com/) is an open-source feature flag and remote config service. Self-hosted via Docker, it provides a web dashboard for managing flags and a REST API + SDKs for evaluation. Key Flagsmith concepts:

| Concept | Description |
|---------|-------------|
| **Organisation** | Top-level container. Holds projects. |
| **Project** | Groups environments and features. |
| **Environment** | Isolated flag configuration (e.g., "Development", "Production"). Each has its own API key. |
| **Feature** | A named flag with enabled/disabled state and optional value. |
| **Identity** | A unique user/entity. Can have individual flag overrides. |
| **Segment** | A group of identities matching a rule (e.g., "beta_users", "enterprise_plan"). |
| **Trait** | A key-value attribute on an identity used for segment matching. |

Flagsmith supports two evaluation modes:
- **Remote evaluation** (default): Every flag check hits the Flagsmith API. Simple but adds latency.
- **Local evaluation** (Server-Side SDK): Downloads the entire environment config and evaluates locally. Zero network latency per evaluation but requires periodic refresh.

### Flag Patterns in Production

**Kill Switch**: A boolean flag that disables an entire feature instantly. When an incident occurs, flip the flag off in the dashboard -- no deployment needed. Every high-risk feature should ship behind a kill switch.

**Gradual Rollout**: Start at 1%, monitor metrics, increase to 5%, 25%, 50%, 100%. If errors spike at any stage, roll back by decreasing the percentage. The flag system uses consistent hashing on user IDs for deterministic bucketing.

**A/B Experiment**: A multivariate flag with "control" and "treatment" variants assigned by percentage. Combine with analytics to measure conversion rates per variant. OpenFeature's evaluation details include the `variant` field for logging which group each user is in.

**Canary Release**: Similar to gradual rollout, but targets a specific segment (e.g., internal employees, beta testers) before broader release. Uses evaluation context attributes for targeting.

**Multi-Service Flag Propagation**: In microservices, the flag evaluation context must propagate across service boundaries. The common pattern is to include flag context (user ID, attributes) in HTTP headers, so downstream services can evaluate flags consistently for the same user.

### Alternatives and Ecosystem

| Tool | Type | Key Differentiator |
|------|------|--------------------|
| **LaunchDarkly** | SaaS | Enterprise-grade, real-time streaming, rich targeting |
| **Unleash** | Open-source | Self-hosted, simple UI, good OSS community |
| **Split.io** | SaaS | Strong experimentation/analytics integration |
| **flagd** | Open-source | Lightweight OpenFeature-native flag daemon |
| **Flagsmith** | Open-source | Full platform (flags + remote config + segments), self-hosted or SaaS |
| **GrowthBook** | Open-source | Experiment-focused, Bayesian statistics built in |

**Trade-offs**: SaaS solutions (LaunchDarkly, Split) offer zero operational overhead but cost money and create vendor dependency. Self-hosted solutions (Flagsmith, Unleash) require infrastructure management but give full control over data. OpenFeature mitigates vendor lock-in for any choice.

## Description

Build a **Feature Flag Playground** with two FastAPI services backed by self-hosted Flagsmith, demonstrating the full spectrum of feature flag patterns through the vendor-agnostic OpenFeature SDK:

```
                    +-----------------+
                    |   Flagsmith UI  |  (http://localhost:8000)
                    |  (Dashboard)    |  -- manage flags, segments, identities
                    +--------+--------+
                             |
                    +--------+--------+
                    |   PostgreSQL    |  (Flagsmith backend)
                    +-----------------+
                             |
          +------------------+------------------+
          |                                     |
  +-------+-------+                   +---------+---------+
  | Product API   |                   | Recommendation    |
  | (port 8001)   |                   | Service (8002)    |
  +-------+-------+                   +---------+---------+
          |                                     |
          +------ HTTP header propagation ------+
          |                                     |
  +-------+-------+                   +---------+---------+
  | OpenFeature   |                   | OpenFeature       |
  | + Flagsmith   |                   | + Flagsmith       |
  | Provider      |                   | Provider          |
  +---------------+                   +-------------------+
```

### What you'll learn

1. **OpenFeature SDK basics** -- Provider registration, client creation, flag evaluation API
2. **Boolean flags** -- Simple on/off toggle (kill switch for maintenance mode)
3. **Multivariate flags** -- String/JSON values (algorithm selection, UI configuration)
4. **Percentage-based rollout** -- Gradual feature exposure with consistent bucketing
5. **User-targeted flags** -- Evaluation context with user attributes for segment targeting
6. **Kill switch pattern** -- Instant feature disable without deployment
7. **A/B experiment** -- Multivariate flag with variant tracking for experiments
8. **OpenFeature hooks** -- Logging, telemetry, and validation via lifecycle callbacks
9. **Multi-service propagation** -- Flag context in HTTP headers across services

## Instructions

### Phase 1: Infrastructure Setup (~10 min)

1. Start Flagsmith, PostgreSQL, and the task processor with `docker compose up -d flagsmith-postgres flagsmith flagsmith-task-processor`
2. Wait for Flagsmith to be ready (health check), then visit `http://localhost:8000`
3. Create an account at `http://localhost:8000/signup` (first user becomes admin)
4. Run the seed script to programmatically create the project, features, and segments: `uv run scripts/seed_flagsmith.py`
5. The seed script outputs the environment API key -- the app services use this
6. Key question: Why do feature flag systems separate "environments" (dev/staging/prod)? What would go wrong if all environments shared the same flag state?

### Phase 2: OpenFeature Provider Setup (~10 min)

1. Review `src/flag_client.py` -- the shared OpenFeature + Flagsmith provider setup
2. **User implements:** `create_openfeature_client()` -- Initialize the Flagsmith Python SDK, create an OpenFeature `FlagsmithProvider`, register it with the OpenFeature API, and return a client
3. **User implements:** `build_evaluation_context()` -- Build an `EvaluationContext` from a user dict with `targeting_key` and attributes (plan, country, beta_tester)
4. Key question: Why does OpenFeature separate "provider" from "client"? What does this abstraction enable?

### Phase 3: Boolean Flags & Kill Switch (~15 min)

1. Review `src/product_api.py` -- the Product API service skeleton
2. **User implements:** `check_maintenance_mode()` -- Evaluate a boolean flag `maintenance_mode` to block all requests when enabled. This is the kill switch pattern: in production, you flip this flag in the Flagsmith dashboard and the API immediately returns 503 without redeployment.
3. **User implements:** `get_products()` endpoint -- Evaluate `new_search_enabled` boolean flag to decide whether to use the new search algorithm or the legacy one. This demonstrates a release toggle: deploy the new code behind a flag, enable it when ready.
4. Test: Toggle `maintenance_mode` on/off in the Flagsmith dashboard and observe the API behavior change in real-time.
5. Key question: Why should kill switches default to "off" (feature enabled) rather than "on"?

### Phase 4: Multivariate Flags & Remote Config (~15 min)

1. **User implements:** `get_recommendation_algorithm()` -- Evaluate a multivariate string flag `recommendation_algorithm` that returns one of: "collaborative", "content_based", "hybrid". This is a remote config pattern: change the algorithm without redeployment.
2. **User implements:** `get_ui_config()` endpoint -- Evaluate a JSON flag `ui_config` that returns layout/theme configuration. This demonstrates dynamic UI configuration without code changes.
3. Test: Change the flag value in Flagsmith dashboard and see the API return different algorithms.
4. Key question: What is the difference between a feature flag and remote configuration? Are they the same thing?

### Phase 5: Percentage Rollout & A/B Experiment (~20 min)

1. **User implements:** `get_checkout_flow()` -- Evaluate a multivariate flag `checkout_experiment` with percentage-based variant assignment. Uses the user's targeting key (user ID) for consistent bucketing -- the same user always sees the same variant.
2. **User implements:** `get_pricing_tier()` -- Evaluate a flag with segment-based targeting. Users with `plan: "enterprise"` get one price, `plan: "starter"` gets another. This demonstrates user-targeted flags via evaluation context attributes.
3. Test: Call the endpoint with different user IDs and observe consistent variant assignment. Verify that the same user always gets the same variant (sticky bucketing).
4. Key question: Why is consistent bucketing essential for A/B experiments? What would happen if users bounced between variants?

### Phase 6: OpenFeature Hooks (~15 min)

1. Review `src/hooks.py` -- hook skeletons
2. **User implements:** `LoggingHook` -- Logs every flag evaluation with flag key, value, and user context. Uses the `after` lifecycle stage. This is how you build observability into your flag system.
3. **User implements:** `MetricsHook` -- Tracks flag evaluation counts and latencies in an in-memory counter. Uses `before` (start timer) and `after` (record duration). In production, this would push to Prometheus/Datadog.
4. **User implements:** Register hooks at the client level
5. Key question: Hooks run on every flag evaluation. What is the performance impact? How do you mitigate it?

### Phase 7: Multi-Service Flag Propagation (~15 min)

1. Review `src/recommendation_service.py` -- the Recommendation Service skeleton
2. **User implements:** `propagate_flag_context()` middleware -- Extract user context from incoming HTTP headers (`X-User-Id`, `X-User-Plan`, etc.) and build an evaluation context. This ensures the downstream service evaluates flags for the same user as the upstream service.
3. **User implements:** `forward_flag_context()` -- When the Product API calls the Recommendation Service, include the user's flag context in HTTP headers. This is the propagation pattern.
4. **User implements:** `get_recommendations()` endpoint in the Recommendation Service -- Uses the propagated context to evaluate flags locally, ensuring consistent flag values across services for the same request.
5. Test: Make a request to the Product API that internally calls the Recommendation Service. Verify both services see the same flag values for the same user.
6. Key question: How does this compare to distributed tracing context propagation (OpenTelemetry)? Could you combine them?

### Phase 8: End-to-End Testing (~10 min)

1. Start all services: `docker compose up --build`
2. Run the test script: `uv run scripts/test_flags.py`
3. Toggle flags in the Flagsmith dashboard (`http://localhost:8000`) and re-run tests to observe behavior changes
4. Experiment: Create a new segment in Flagsmith that targets users by country, then test with different evaluation contexts

## Motivation

- **Industry standard**: Feature flags are a core practice in modern continuous delivery (used by Netflix, GitHub, Meta, Uber). Every production backend engineer needs to understand flag-driven releases.
- **OpenFeature is the future**: As CNCF's feature flag standard, OpenFeature is becoming the vendor-agnostic API for flags -- analogous to OpenTelemetry for observability. Learning it now is a strategic investment.
- **Complements existing practices**: Builds on Docker Compose (005), API design (016), and leads into observability (007a) and resilience patterns (052). Kill switches are a key resilience mechanism.
- **Practical AutoScheduler.AI relevance**: Feature flags enable safe rollout of optimization algorithm changes, A/B testing of scheduling heuristics, and instant rollback of problematic features.

## Commands

All commands are run from the `practice_080_feature_flags/` folder root.

### Phase 1: Infrastructure

| Command | Description |
|---------|-------------|
| `docker compose up -d flagsmith-postgres flagsmith flagsmith-task-processor` | Start Flagsmith backend (PostgreSQL + API + task processor) |
| `docker compose ps` | Check status of all containers |
| `docker compose logs flagsmith` | View Flagsmith API logs |
| `docker compose logs flagsmith-task-processor` | View Flagsmith task processor logs |

### Phase 1: Flagsmith Setup

| Command | Description |
|---------|-------------|
| Open `http://localhost:8000/signup` in browser | Create admin account (first-time setup) |
| Open `http://localhost:8000` in browser | Flagsmith dashboard -- manage flags, segments, identities |
| `uv run scripts/seed_flagsmith.py` | Seed Flagsmith with project, environment, features, and segments. Outputs the environment API key. |

### Phase 2-6: Run Services Locally

| Command | Description |
|---------|-------------|
| `uv run scripts/seed_flagsmith.py` | Re-run seed if needed (idempotent) |
| `uv run src/product_api.py` | Start Product API service on port 8001 |
| `uv run src/recommendation_service.py` | Start Recommendation Service on port 8002 |

### Phase 7: Run All via Docker Compose

| Command | Description |
|---------|-------------|
| `docker compose up --build` | Build and start all services (Flagsmith + Product API + Recommendation Service) |
| `docker compose up -d` | Start all services in background |
| `docker compose logs -f product-api` | Follow Product API logs |
| `docker compose logs -f recommendation-service` | Follow Recommendation Service logs |

### Phase 8: End-to-End Testing

| Command | Description |
|---------|-------------|
| `uv run scripts/test_flags.py` | Run end-to-end tests against all endpoints |
| `curl http://localhost:8001/health` | Product API health check |
| `curl http://localhost:8001/products -H "X-User-Id: user-1"` | Get products (flag-controlled search) |
| `curl http://localhost:8001/products -H "X-User-Id: user-1" -H "X-User-Plan: enterprise"` | Get products with enterprise plan context |
| `curl http://localhost:8001/checkout -H "X-User-Id: user-1"` | Get checkout flow (A/B experiment) |
| `curl http://localhost:8001/recommendations -H "X-User-Id: user-1" -H "X-User-Plan: starter"` | Get recommendations (cross-service propagation) |
| `curl http://localhost:8002/recommendations -H "X-User-Id: user-1"` | Direct call to Recommendation Service |

### Inspection & Cleanup

| Command | Description |
|---------|-------------|
| Open `http://localhost:8000` in browser | Flagsmith dashboard -- toggle flags, inspect identities, manage segments |
| `docker compose down` | Stop and remove all containers |
| `docker compose down -v` | Stop containers and delete PostgreSQL data volume |
| `python clean.py` | Full cleanup: Docker down, remove caches and generated files |

## References

- [Martin Fowler: Feature Toggles](https://martinfowler.com/articles/feature-toggles.html)
- [OpenFeature Specification](https://openfeature.dev/specification/)
- [OpenFeature Python SDK](https://openfeature.dev/docs/reference/sdks/server/python/)
- [OpenFeature Concepts: Providers](https://openfeature.dev/docs/reference/concepts/provider/)
- [OpenFeature Concepts: Hooks](https://openfeature.dev/docs/reference/concepts/hooks/)
- [OpenFeature Concepts: Evaluation Context](https://openfeature.dev/docs/reference/concepts/evaluation-context/)
- [Flagsmith Documentation](https://docs.flagsmith.com/)
- [Flagsmith Docker Deployment](https://docs.flagsmith.com/deployment-self-hosting/hosting-guides/docker)
- [Flagsmith Python SDK](https://docs.flagsmith.com/clients/server-side)
- [Flagsmith OpenFeature Provider (Python)](https://github.com/Flagsmith/flagsmith-openfeature-provider-python)
- [Flagsmith REST API](https://docs.flagsmith.com/clients/rest/)
- [LaunchDarkly: Feature Flag Best Practices](https://launchdarkly.com/blog/release-management-flags-best-practices/)
- [Octopus: 12 Commandments of Feature Flags](https://octopus.com/devops/feature-flags/feature-flag-best-practices/)
- [Flagsmith: Deployment Strategies](https://www.flagsmith.com/blog/deployment-strategies)

## State

`not-started`
