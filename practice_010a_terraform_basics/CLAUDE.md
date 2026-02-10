# Practice 010a: Terraform -- Basics & Local Resources

## Technologies

- **Terraform** -- Infrastructure as Code (IaC) tool by HashiCorp
- **HCL** -- HashiCorp Configuration Language (declarative DSL for infrastructure)
- **kreuzwerker/docker** -- Terraform provider for managing Docker resources locally
- **Docker Engine** -- Container runtime (already installed from previous practices)

## Stack

- HCL (Terraform configuration language)
- Docker provider (no cloud credentials needed)

## Description

Build **local Docker infrastructure entirely through Terraform**: pull images, create networks, volumes, and containers -- all declared in `.tf` files. This teaches Terraform's core mechanics -- providers, resources, variables, outputs, state, plan/apply/destroy lifecycle -- without needing any cloud account.

### What you'll learn

1. **HCL syntax** -- blocks, arguments, attributes, expressions, string interpolation
2. **Terraform workflow** -- `init` (download providers) -> `plan` (preview changes) -> `apply` (create resources) -> `destroy` (tear down)
3. **Provider configuration** -- declaring and configuring the Docker provider
4. **Resources** -- `docker_image`, `docker_container`, `docker_network`, `docker_volume`
5. **Variables & outputs** -- input parameterization (`variable`), computed values (`output`), tfvars files
6. **Local values** -- `locals` for DRY computed expressions
7. **State** -- how `terraform.tfstate` tracks real infrastructure, why it matters
8. **Resource dependencies** -- implicit (reference-based) and explicit (`depends_on`)
9. **Data sources** -- reading existing Docker resources without managing them
10. **Lifecycle management** -- `create_before_destroy`, `prevent_destroy`, `ignore_changes`

## Instructions

### Exercise 1: Hello Terraform (`01_hello/`) -- ~15 min

**Goal:** Minimal Terraform config. Pull an nginx image and run a container.

1. Read `main.tf` -- understand the `terraform {}`, `provider {}`, and `resource {}` blocks
2. **User implements:** The `required_providers` block pointing to `kreuzwerker/docker`
3. **User implements:** A `docker_image` resource that pulls `nginx:alpine`
4. **User implements:** A `docker_container` resource that runs it on port 8080->80
5. Run `terraform init`, `terraform plan`, `terraform apply`
6. Visit `http://localhost:8080` to verify nginx is running
7. Run `terraform destroy` to clean up
8. Key question: What does `terraform.tfstate` contain after apply? Why is it dangerous to edit manually?

### Exercise 2: Variables & Outputs (`02_variables/`) -- ~20 min

**Goal:** Parameterize the configuration. Same nginx container, but configurable.

1. Read `variables.tf` -- understand `variable` blocks (type, default, description, validation)
2. **User implements:** Variables for container name, image tag, and external port
3. **User implements:** A validation rule on the port variable (must be 1024-65535)
4. **User implements:** `outputs.tf` -- expose container ID, IP address, and the access URL
5. **User implements:** Reference the variables in `main.tf` using `var.<name>` syntax
6. Apply with defaults, then override via `terraform.tfvars` and `-var` CLI flag
7. Run `terraform output` to see computed values
8. Key question: What is the variable precedence order? (defaults < tfvars < env < CLI)

### Exercise 3: Locals & Multiple Resources (`03_locals/`) -- ~20 min

**Goal:** Deploy two containers (nginx + Redis) on a shared Docker network using locals.

1. Read `main.tf` -- understand how `locals {}` block avoids repetition
2. **User implements:** A `locals` block with a common naming prefix and labels map
3. **User implements:** A `docker_network` resource for inter-container communication
4. **User implements:** Two `docker_container` resources attached to the network
5. **User implements:** Use `local.<name>` references for consistent naming
6. Apply and verify both containers are on the same network (`docker network inspect`)
7. Key question: How does Terraform know to create the network before the containers? (implicit dependency via resource references)

### Exercise 4: Data Sources & Lifecycle (`04_data_lifecycle/`) -- ~15 min

**Goal:** Use data sources to read existing Docker state. Explore lifecycle rules.

1. Read `main.tf` -- understand `data` blocks vs `resource` blocks
2. **User implements:** A `data "docker_network"` that reads the default `bridge` network
3. **User implements:** A `docker_container` with `lifecycle { create_before_destroy = true }`
4. Modify the container config and re-apply -- observe zero-downtime replacement
5. **User implements:** Add `prevent_destroy = true`, try to destroy, observe the error
6. Key question: When would you use `ignore_changes`? (hint: external modifications)

### Exercise 5: Full Stack (`05_full_stack/`) -- ~20 min

**Goal:** Combine everything: a multi-container stack with nginx reverse proxy + two app containers.

1. Read through all `.tf` files -- this ties together all previous concepts
2. **User implements:** The complete `docker_network` resource
3. **User implements:** Two `docker_container` "app" resources using `count` or `for_each`
4. **User implements:** An nginx container with volume-mounted config
5. **User implements:** All outputs (container IPs, access URLs, network ID)
6. Apply, verify the stack, then destroy
7. Key question: What would happen if you deleted `terraform.tfstate` and ran apply again? (hint: Terraform would try to create duplicates)

## Motivation

- **Industry standard IaC**: Terraform is the most widely adopted infrastructure-as-code tool, used across AWS, GCP, Azure, and on-prem environments
- **DevOps literacy**: Understanding declarative infrastructure management is essential for any backend/platform engineer
- **Complementary to Docker/K8s**: Terraform orchestrates the infrastructure that containers run on -- different abstraction layer than Docker Compose or Kubernetes
- **Career demand**: Terraform/IaC skills consistently appear in senior backend, platform, and SRE job postings
- **Local-first learning**: The Docker provider lets you learn all Terraform concepts without cloud costs or credentials

## References

- [Terraform Documentation](https://developer.hashicorp.com/terraform/docs)
- [Terraform Docker Get Started Tutorial](https://developer.hashicorp.com/terraform/tutorials/docker-get-started)
- [kreuzwerker/docker Provider Docs](https://registry.terraform.io/providers/kreuzwerker/docker/latest/docs)
- [HCL Syntax Specification](https://developer.hashicorp.com/terraform/language/syntax/configuration)
- [Terraform Variables Guide](https://developer.hashicorp.com/terraform/language/values/variables)
- [Terraform State Documentation](https://developer.hashicorp.com/terraform/language/state)
- [Terraform Style Guide](https://developer.hashicorp.com/terraform/language/style)

## Commands

Each exercise lives in its own subdirectory. Run commands from within the exercise folder (e.g., `cd 01_hello`).

### Exercise 1: Hello Terraform (`01_hello/`)

| Command | Description |
|---------|-------------|
| `terraform init` | Download the kreuzwerker/docker provider plugin |
| `terraform plan` | Preview resources to be created (image + container) |
| `terraform apply` | Create the nginx image and container (prompts "yes") |
| `terraform apply -auto-approve` | Create resources without confirmation prompt |
| `curl http://localhost:8080` | Verify nginx is running |
| `terraform destroy` | Tear down the container and remove the image |

### Exercise 2: Variables & Outputs (`02_variables/`)

| Command | Description |
|---------|-------------|
| `terraform init` | Download the Docker provider plugin |
| `terraform plan` | Preview with defaults from `terraform.tfvars` |
| `terraform apply` | Create the parameterized nginx container |
| `terraform apply -var="external_port=9090"` | Override a variable via CLI flag |
| `terraform apply -var="container_name=custom-nginx" -var="image_tag=latest"` | Override multiple variables via CLI |
| `terraform output` | Show all computed output values (container ID, name, URL) |
| `terraform output access_url` | Show a single output value |
| `terraform destroy` | Tear down all resources |

### Exercise 3: Locals & Multiple Resources (`03_locals/`)

| Command | Description |
|---------|-------------|
| `terraform init` | Download the Docker provider plugin |
| `terraform plan` | Preview network, images, and containers to be created |
| `terraform apply` | Create the shared network, nginx, and Redis containers |
| `docker network inspect tf-practice-network` | Verify both containers are on the shared network |
| `terraform output` | Show network name, nginx URL, and Redis endpoint |
| `terraform destroy` | Tear down containers, images, and network |

### Exercise 4: Data Sources & Lifecycle (`04_data_lifecycle/`)

| Command | Description |
|---------|-------------|
| `terraform init` | Download the Docker provider plugin |
| `terraform plan` | Preview resources (reads bridge network data source) |
| `terraform apply` | Create the container with lifecycle rules |
| `terraform apply -var="external_port=8082"` | Change the port to trigger `create_before_destroy` replacement |
| `terraform apply -var="container_name=renamed-demo"` | Change name to test `ignore_changes` (if configured) |
| `terraform destroy` | Tear down resources (will fail if `prevent_destroy = true` is active) |

### Exercise 5: Full Stack (`05_full_stack/`)

| Command | Description |
|---------|-------------|
| `terraform init` | Download the Docker provider plugin |
| `terraform plan` | Preview the full stack (network, images, app containers, proxy) |
| `terraform apply` | Deploy the entire stack |
| `terraform apply -var="app_instances=3"` | Deploy with a different number of app instances |
| `terraform apply -var="proxy_port=9090"` | Deploy with a custom proxy port |
| `curl http://localhost:8001` | Test app-0 directly ("Hello from app-0") |
| `curl http://localhost:8002` | Test app-1 directly ("Hello from app-1") |
| `curl http://localhost:8080` | Test the nginx reverse proxy (load-balances between apps) |
| `terraform output` | Show network ID, app URLs, proxy URL, container names |
| `terraform destroy` | Tear down the entire stack |

### General Terraform Commands (any exercise)

| Command | Description |
|---------|-------------|
| `terraform fmt` | Auto-format `.tf` files in the current directory |
| `terraform validate` | Check syntax and internal consistency of config |
| `terraform state list` | List all resources tracked in state |
| `terraform state show <address>` | Show details of a specific resource (e.g., `docker_container.web`) |

## State

`not-started`
