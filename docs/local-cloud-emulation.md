# Local Cloud Service Emulation — Complete Reference

Comprehensive guide to running cloud services locally using Docker, without real cloud accounts.

---

## Overview

There is **no single tool that emulates everything**. The ecosystem breaks down into:

1. **Multi-service emulators** — LocalStack (AWS), no equivalent for GCP/Azure
2. **Official provider emulators** — GCP and Azure offer per-service emulators
3. **Open-source replacements** — Standalone tools with cloud-compatible APIs
4. **Frameworks** — Dapr, Testcontainers for abstracting cloud dependencies

---

## 1. Multi-Service Emulators

### LocalStack (AWS)

The only viable multi-service AWS emulator. See [localstack.md](./localstack.md) for full details.

- 80+ services (Community), 130+ (Pro)
- Single Docker container on port 4566
- CLI wrappers: `awslocal`, `tflocal`
- Community is free but **no persistence** (data lost on restart)

### GCP — No Equivalent

Google provides individual emulators per service (see section 2). No unified LocalStack-like tool exists.

### Azure — No Equivalent

Microsoft provides Azurite (Storage) and Cosmos DB Emulator. No unified multi-service emulator.

### Cloud-Agnostic — Dapr

See section 5 (Frameworks).

---

## 2. Official Cloud Provider Emulators

### AWS Official

| Service | Tool | Docker Image | Notes |
|---------|------|-------------|-------|
| DynamoDB | DynamoDB Local | `amazon/dynamodb-local` | Exact API, in-memory or file-backed |
| Lambda | AWS SAM CLI | N/A (uses Docker) | `sam local invoke`, simulates events (S3, SQS, Kinesis, SNS) |

AWS officially recognizes LocalStack for comprehensive multi-service testing (VS Code integration, 2025).

### GCP Official

All available via `gcloud beta emulators` or Docker:

| Service | Command / Image | Docker Support | Quality |
|---------|----------------|---------------|---------|
| Pub/Sub | `gcloud beta emulators pubsub start` | Yes | Production-grade |
| Firestore | `gcloud beta emulators firestore start` | Yes | Production-grade |
| Datastore | `gcloud beta emulators datastore start` | Yes | Production-grade |
| Bigtable | `gcloud beta emulators bigtable start` | Yes | Production-grade |
| Spanner | `gcloud beta emulators spanner start` | Yes | Production-grade |

**Firebase Local Emulator Suite** covers: Firestore, Pub/Sub, Cloud Functions, Authentication, Storage — all in one.

**Testcontainers GCloud module** provides programmatic Docker management for all the above.

### Azure Official

| Service | Tool | Docker Image | Notes |
|---------|------|-------------|-------|
| Blob/Queue/Table Storage | Azurite | `mcr.microsoft.com/azure-storage/azurite` | Lightweight Node.js, exact API |
| Cosmos DB | Cosmos DB Emulator | Docker container | Heavier, requires specific config |
| Service Bus | Docker Compose setup | Needs SQL Server backend | More complex setup |

---

## 3. Open-Source Replacements by Category

### Object Storage (S3-compatible)

| Tool | Language | Docker Image | Size | Best For |
|------|----------|-------------|------|----------|
| **MinIO** | Go | `minio/minio` | ~100MB | Pure S3 workloads, web UI, production-grade |
| **SeaweedFS** | Go | `chrislusf/seaweedfs` | Light | Distributed blob/file storage |
| **Garage** | Rust | `dxflrs/garage` | Light | S3-compatible, static website hosting |

**MinIO vs LocalStack S3:** MinIO excels at GETs for medium objects. LocalStack better for PUTs and when you need other AWS services alongside S3.

```bash
# MinIO quickstart
docker run -p 9000:9000 -p 9001:9001 \
  minio/minio server /data --console-address ":9001"
# Web UI at http://localhost:9001 (minioadmin/minioadmin)
```

### Message Queues

| Tool | Protocol | Docker Image | Latency | Best For |
|------|----------|-------------|---------|----------|
| **Redpanda** | Kafka API | `redpandadata/redpanda` | 10x better p99.99 than Kafka | Kafka replacement, no Zookeeper |
| **NATS** | NATS/JetStream | `nats` | Sub-ms | Tiny footprint, real-time |
| **ElasticMQ** | SQS API | `softwaremill/elasticmq` | Fast | Drop-in SQS replacement (~30MB) |
| **RabbitMQ** | AMQP | `rabbitmq:management` | Low | Traditional messaging, mature |
| **Redis Streams** | Redis | `redis` | Sub-ms | In-memory, low latency |

**Redpanda vs Kafka:** Redpanda is Kafka API-compatible, written in C++, no JVM, no Zookeeper. Single binary. Recommended for local dev and learning Kafka patterns without the complexity.

```bash
# Redpanda quickstart (Kafka-compatible)
docker run -p 9092:9092 -p 8081:8081 -p 8082:8082 -p 9644:9644 \
  redpandadata/redpanda start --smp 1 --memory 1G --overprovisioned

# ElasticMQ quickstart (SQS-compatible)
docker run -p 9324:9324 -p 9325:9325 softwaremill/elasticmq-native
```

### Pub/Sub & Event Streaming

| Tool | Compatibility | Docker | Best For |
|------|--------------|--------|----------|
| **Redpanda** | Kafka API | Yes | Streaming with Kafka clients |
| **NATS JetStream** | NATS | Yes | Lightweight pub/sub + persistence |
| **Apache Pulsar** | Pulsar API | Yes | Multi-tenancy, geo-replication |
| **GCP Pub/Sub Emulator** | GCP Pub/Sub API | Yes | Exact GCP API practice |

### NoSQL Databases

| Tool | Compatibility | Docker Image | Best For |
|------|--------------|-------------|----------|
| **DynamoDB Local** | DynamoDB API | `amazon/dynamodb-local` | Exact AWS API |
| **ScyllaDB Alternator** | DynamoDB API | `scylladb/scylla` | High-perf DynamoDB alternative |
| **ScyllaDB** | Cassandra API | `scylladb/scylla` | 10x faster than Cassandra |
| **MongoDB** | MongoDB | `mongo` | Document store, rich queries |
| **CockroachDB** | PostgreSQL wire | `cockroachdb/cockroach` | NewSQL, strong consistency |

```bash
# DynamoDB Local
docker run -p 8000:8000 amazon/dynamodb-local

# ScyllaDB with DynamoDB-compatible Alternator API
docker run -p 9042:9042 -p 8000:8000 scylladb/scylla --alternator-port=8000
```

### Key-Value Stores

| Tool | Docker Image | Best For |
|------|-------------|----------|
| **Redis** | `redis` | Caching, pub/sub, streams |
| **Memcached** | `memcached` | Pure caching |
| **etcd** | `quay.io/coreos/etcd` | Distributed config, K8s backing store |

### Serverless / Functions

| Tool | Backed By | Complexity | Best For |
|------|-----------|-----------|----------|
| **Knative** | Google/IBM/Red Hat (CNCF) | High (needs K8s) | Industry standard, multi-language |
| **OpenFaaS** | Community | Medium (Docker Swarm or K8s) | Simpler than Knative |
| **Fission** | Platform9 | Low | Fastest cold-starts, no Dockerfiles needed |
| **AWS SAM Local** | AWS | Low | Lambda-specific, simulates events |

Knative has the strongest ecosystem. Fission is best for quick setup. Both require Kubernetes (use k3d).

### API Gateway / Reverse Proxy

| Tool | Language | Docker Image | Complexity | Best For |
|------|----------|-------------|-----------|----------|
| **Traefik** | Go | `traefik` | Low | Local dev, auto-discovery via Docker labels |
| **Kong** | Lua/Go | `kong` | Medium | Plugin ecosystem, production-like |
| **Envoy** | C++ | `envoyproxy/envoy` | High | Service mesh, high-performance |

**Traefik recommended for local dev** — single binary, auto-discovers Docker containers, no external DB. Kong is heavier but more feature-rich.

### Container Orchestration (Local Kubernetes)

| Tool | Mechanism | Memory | Startup | Best For |
|------|-----------|--------|---------|----------|
| **k3d** | k3s in Docker | <512MB | Fastest | Most local dev, lightweight |
| **Kind** | K8s in Docker | ~1GB | Fast | CI/CD pipelines, multi-node |
| **Minikube** | VM/Docker | ~2GB | Slower | Full K8s features, tutorials |
| **MicroK8s** | Snap | ~1GB | Fast | Ubuntu/Canonical environments |

**Recommendation:** Start with **k3d** for speed, use **Kind** for CI, **Minikube** for full K8s feature exploration.

```bash
# k3d quickstart
k3d cluster create dev --port 8080:80@loadbalancer

# Kind quickstart
kind create cluster --name dev
```

### Identity / Authentication

| Tool | Scope | Docker Image | Best For |
|------|-------|-------------|----------|
| **Keycloak** | Full IAM (auth, authz, users, OIDC, SAML) | `quay.io/keycloak/keycloak` | All-in-one, web UI |
| **Ory Hydra** | OAuth2/OIDC only (headless) | `oryd/hydra` | Lightweight, modular microservices |

Keycloak for all-in-one. Ory stack (Hydra + Kratos + Keto) for modular approach.

### Monitoring / Observability

All run natively in Docker — no emulation needed:

| Tool | Purpose | Docker Image |
|------|---------|-------------|
| **Prometheus** | Metrics collection | `prom/prometheus` |
| **Grafana** | Dashboards | `grafana/grafana` |
| **Jaeger** | Distributed tracing | `jaegertracing/all-in-one` |
| **OpenTelemetry Collector** | Telemetry pipeline | `otel/opentelemetry-collector` |

---

## 4. Infrastructure as Code (Local)

### Terraform

- **Docker provider** — manages Docker containers/networks/volumes as resources
- **`tflocal`** — wrapper that auto-points Terraform to LocalStack
- Integration with all local tools (LocalStack, MinIO, etc.)

### Pulumi

- Supports 150+ providers including Docker
- Native testing with pytest/xUnit/Go tests
- Uses real programming languages (TypeScript, Python, Go, .NET, Java)
- Better for unit testing (mocks external calls)

---

## 5. Frameworks

### Testcontainers

Library for programmatic Docker container management in tests. Available for Java, Python, Go, .NET, Node.js.

Pre-built modules for: PostgreSQL, MySQL, MongoDB, Redis, RabbitMQ, Azurite, Cosmos DB, GCP emulators (Pub/Sub, Firestore, Bigtable, Spanner).

Ideal for **integration testing** without manual Docker Compose setup.

```python
# Python example
from testcontainers.redis import RedisContainer

with RedisContainer() as redis:
    client = redis.get_client()
    client.set("key", "value")
```

### Dapr (Distributed Application Runtime)

CNCF runtime for building distributed apps. Provides sidecar APIs for:
- Service-to-service invocation
- Pub/sub messaging
- State management
- Secrets
- Bindings (input/output)

Abstracts away specific cloud services — write once, run on any cloud or locally.

```bash
dapr init    # Sets up local environment with Redis + Zipkin
dapr run --app-id myapp -- python app.py
```

96% of developers report time savings (2025 State of Dapr Report).

---

## 6. Comparison Matrix — Setup Complexity vs Features

```
High Features │                        ┌──LocalStack Pro
              │          Knative ──────┤
              │          Kong          │  Keycloak
              │                        │
Medium        │  Redpanda      k3d     │
Features      │          Traefik       │
              │  MinIO                 │  Azurite
              │  NATS                  │
Low Features  │  ElasticMQ             └──DynamoDB Local
              │
              └──────────────────────────────────────
                Low        Medium          High
                    Setup Complexity
```

---

## 7. Recommendations for This Project

Given: 60–120 min guided practices, Docker-based, no real cloud accounts.

### Per-Practice Tool Selection

| Practice | Recommended Tool | Why (over alternatives) |
|----------|-----------------|------------------------|
| 002 Cloud Pub/Sub | GCP Emulator OR LocalStack SQS/SNS | Official emulator for GCP parity; LocalStack for AWS parity |
| 003a/b Kafka | **Redpanda** | Kafka API-compatible, no Zookeeper, simpler Docker setup |
| 005 Docker Compose | Multiple (Traefik + Redis + PostgreSQL + app) | Real multi-service orchestration |
| 006a/b Kubernetes | **k3d** | Fastest startup, lowest resources, Docker-native |
| 007a/b OpenTelemetry | Prometheus + Grafana + Jaeger | All native Docker, no emulation needed |
| 008 Vector Databases | Qdrant | Native Docker, purpose-built |
| 010a/b Terraform | **LocalStack + tflocal** | Terraform managing real (emulated) AWS resources |

### Key Insight

**LocalStack is the best choice when you need multiple AWS services interacting** (e.g., Lambda triggered by SQS writing to DynamoDB). For single-service practices, **standalone tools are better** (Redpanda > LocalStack Kafka, MinIO > LocalStack S3, etc.) because they provide the real tool's CLI, monitoring, and behavior.

---

## Sources

- [LocalStack GitHub](https://github.com/localstack/localstack)
- [GCP Pub/Sub Emulator](https://cloud.google.com/pubsub/docs/emulator)
- [Firebase Local Emulator Suite](https://firebase.google.com/docs/emulator-suite)
- [Microsoft Azurite](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azurite)
- [Azure Cosmos DB Emulator](https://learn.microsoft.com/en-us/azure/cosmos-db/emulator)
- [AWS DynamoDB Local](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/DynamoDBLocal.DownloadingAndRunning.html)
- [AWS SAM Local](https://aws.amazon.com/blogs/aws/new-aws-sam-local-beta-build-and-test-serverless-applications-locally/)
- [S3 Mocking Tools Comparison (LocalStack Blog)](https://blog.localstack.cloud/2024-04-08-exploring-s3-mocking-tools-a-comparative-analysis-of-s3mock-minio-and-localstack/)
- [MinIO vs LocalStack (SingleStore)](https://www.singlestore.com/blog/migrating-from-minio-to-localstack/)
- [2025 Message Broker Comparison](https://medium.com/@BuildShift/kafka-is-old-redpanda-is-fast-pulsar-is-weird-nats-is-tiny-which-message-broker-should-you-32ce61d8aa9f)
- [ElasticMQ GitHub](https://github.com/softwaremill/elasticmq)
- [NATS Documentation](https://docs.nats.io/nats-concepts/overview/compare-nats)
- [Awesome Self-Hosted AWS](https://github.com/fffaraz/awesome-selfhosted-aws)
- [Serverless Frameworks Comparison](https://palark.com/blog/open-source-self-hosted-serverless-frameworks-for-kubernetes/)
- [CNCF Dapr Guide (Dec 2025)](https://www.cncf.io/blog/2025/12/09/building-microservices-the-easy-way-with-dapr/)
- [Dapr Local Development](https://docs.dapr.io/developing-applications/local-development/)
- [Kubernetes API Gateway Comparison](https://www.cloudraft.io/blog/kubernetes-api-gateway-comparison)
- [Local Kubernetes Showdown 2025](https://sanj.dev/post/2025-12-11-ultimate-local-kubernetes-showdown-2025)
- [Minikube vs k3s vs Kind](https://www.automq.com/blog/minikube-vs-k3s-vs-kind-comparison-local-kubernetes-development)
- [ScyllaDB Alternator](https://www.scylladb.com/alternator/)
- [Pulumi vs Terraform](https://spacelift.io/blog/pulumi-vs-terraform)
- [Open Source Auth Providers 2025](https://tesseral.com/guides/open-source-auth-providers-in-2025-best-solutions-for-open-source-auth)
- [Testcontainers](https://testcontainers.com/)
- [Testcontainers GCloud Module](https://java.testcontainers.org/modules/gcloud/)
- [Docker Compose Best Practices 2025](https://release.com/blog/6-docker-compose-best-practices-for-dev-and-prod)
