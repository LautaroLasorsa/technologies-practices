# LocalStack — AWS Local Emulation

AWS services running locally in Docker, for free (Community edition). Current version: **4.12** (early 2025).

100M+ Docker pulls. Acts as a mini-cloud OS intercepting AWS SDK/CLI calls on port **4566**.

---

## Community vs Pro

| Feature | Community (free) | Pro ($50+/mo) |
|---------|-----------------|---------------|
| Core services (S3, SQS, SNS, DynamoDB, Lambda) | Yes | Yes |
| Service count | ~80 | 130+ |
| Persistence | No (data lost on restart) | Yes |
| IAM enforcement | No | Yes |
| Cloud Pods (shared state snapshots) | No | Yes |
| Ephemeral Instances | No | Yes |
| Bedrock (AI) | No | Yes |

Pro requires `LOCALSTACK_AUTH_TOKEN` env var.

---

## Services Available

**Compute:** Lambda, ECS, EKS, EC2
**Storage:** S3, EBS, EFS
**Databases:** DynamoDB, RDS, DocumentDB, Neptune, Redshift, ElastiCache
**Messaging:** SQS, SNS, EventBridge, Kinesis, MSK
**API:** API Gateway (REST/HTTP)
**Orchestration:** Step Functions, CloudFormation
**Monitoring:** CloudWatch
**Security:** IAM, Cognito, Secrets Manager, Systems Manager
**CI/CD:** CodeBuild, CodePipeline
**Data:** Glue, Athena
**AI (Pro):** Bedrock

Full list: https://docs.localstack.cloud/user-guide/aws/feature-coverage/

---

## Installation

Four methods (all expose port 4566):

```bash
# 1. LocalStack CLI (recommended)
# Linux/macOS
curl -Lo localstack-cli.tar.gz https://github.com/localstack/localstack-cli/releases/latest/...
# macOS via Homebrew
brew install localstack
localstack start

# 2. pip
pip install localstack

# 3. Docker directly
docker run -p 4566:4566 -v /var/run/docker.sock:/var/run/docker.sock localstack/localstack

# 4. Docker Compose (see below)
```

---

## Docker Compose Setup

```yaml
services:
  localstack:
    container_name: localstack-main
    image: localstack/localstack
    ports:
      - "127.0.0.1:4566:4566"       # Gateway
      - "127.0.0.1:4510-4559:4510-4559"  # External services
    environment:
      - DEBUG=0
      - PERSISTENCE=1                # Pro only for real persistence
      - SERVICES=s3,sqs,dynamodb     # Optional: limit to specific services
    volumes:
      - "./volume:/var/lib/localstack"
      - "/var/run/docker.sock:/var/run/docker.sock"  # Required for Lambda/ECS
```

**Best practices:**
- Bind to `127.0.0.1` (not `0.0.0.0`)
- Mount Docker socket only if using Lambda or ECS
- Use init scripts at `/etc/localstack/init/ready.d/` for startup automation
- Apps in same Compose use `http://localstack:4566` as endpoint (service name)
- Validate config: `localstack config validate`

---

## CLI Wrapper Tools

All auto-configure `--endpoint-url` to `http://localhost:4566`:

| Tool | Wraps | Install |
|------|-------|---------|
| `awslocal` | AWS CLI | `pip install awscli-local` |
| `tflocal` | Terraform | `pip install terraform-local` |
| `cdklocal` | AWS CDK | GitHub install |
| `samlocal` | AWS SAM CLI | GitHub install |

```bash
# Instead of: aws --endpoint-url=http://localhost:4566 s3 ls
awslocal s3 ls

# Instead of: terraform apply (with manual provider config)
tflocal init && tflocal apply
```

`tflocal` supports `TF_CMD=tofu` env var for OpenTofu.

---

## Terraform Integration

**Option 1 — tflocal (recommended):**
```bash
pip install terraform-local
tflocal init
tflocal apply
# Auto-generates localstack_providers_override.tf
```

**Option 2 — Manual provider config:**
```hcl
provider "aws" {
  access_key                  = "test"
  secret_key                  = "test"
  region                      = "us-east-1"
  s3_use_path_style           = true
  skip_credentials_validation = true
  skip_metadata_api_check     = true

  endpoints {
    s3       = "http://s3.localhost.localstack.cloud:4566"
    sqs      = "http://localhost:4566"
    dynamodb = "http://localhost:4566"
    lambda   = "http://localhost:4566"
    iam      = "http://localhost:4566"
    # ... repeat per service
  }
}
```

Detection: `aws_caller_identity.id == "000000000000"` → LocalStack.

---

## Python (boto3) Usage

```python
import boto3

# Direct endpoint
s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:4566",
    aws_access_key_id="test",
    aws_secret_access_key="test",
)

# Or via environment variable (cleaner)
# export AWS_ENDPOINT_URL=http://localhost:4566
s3 = boto3.client("s3")  # auto-routes to LocalStack
```

In Docker Compose, use `http://localstack:4566` instead of `localhost`.

---

## Persistence (Pro)

Enable: `PERSISTENCE=1`

**Save strategies** (`SNAPSHOT_SAVE_STRATEGY`):

| Strategy | Behavior | Trade-off |
|----------|----------|-----------|
| `SCHEDULED` (default) | Every 15s for modified services | Balanced |
| `ON_REQUEST` | Every state-modifying API call | Max safety, performance hit |
| `ON_SHUTDOWN` | Only at shutdown | Fast, risk of data loss |
| `MANUAL` | API call: `POST /_localstack/state/<service>/save` | Full control |

Data at `/var/lib/localstack/state`. Caveat: snapshots may break across LocalStack versions.

---

## CI/CD — GitHub Actions

```yaml
- name: Start LocalStack
  uses: LocalStack/setup-localstack@v0.2.3
  with:
    image-tag: latest
    install-awslocal: true
    configuration: DEBUG=1
  env:
    LOCALSTACK_AUTH_TOKEN: ${{ secrets.LOCALSTACK_AUTH_TOKEN }}  # Pro only

- name: Run tests
  env:
    AWS_ENDPOINT_URL: http://localhost:4566
  run: pytest
```

---

## Limitations & Gotchas

- **No IAM enforcement** (Community) — tests pass locally, fail in AWS due to permissions
- **No AWS quotas** — app may hit limits only in production
- **Community has no persistence** — data lost on container restart
- **Timing differs** — Lambda cold starts, SQS delays, EventBridge processing ≠ real AWS
- **Not 100% API parity** — check individual service docs for gaps
- **Windows Git Bash** — auto-converts POSIX paths, causing errors
- **Persistence caveats** (Pro) — ports may change across restarts, snapshots not guaranteed across versions

---

## Alternatives

| Tool | Best for | vs LocalStack |
|------|----------|---------------|
| **MinIO** | Production S3 replacement | Single-service; production-grade HA. LocalStack = dev/test multi-service |
| **moto** | Python unit test mocking | Code-level mocking, no network. LocalStack = service-level emulation |
| **ElasticMQ** | SQS-only emulation | Lightweight SQS. LocalStack = comprehensive |

---

## Relevant Practices

- **002 Cloud Pub/Sub** — SQS/SNS via LocalStack instead of Google emulator
- **005 Docker Compose** — LocalStack is itself a multi-service Compose example
- **010a/b Terraform** — `tflocal` pointing to LocalStack for fully local IaC

---

## Links

- https://localstack.cloud/
- https://docs.localstack.cloud/
- https://github.com/localstack/localstack
- https://docs.localstack.cloud/user-guide/aws/feature-coverage/
- https://docs.localstack.cloud/aws/integrations/infrastructure-as-code/terraform/
- https://github.com/localstack/setup-localstack
