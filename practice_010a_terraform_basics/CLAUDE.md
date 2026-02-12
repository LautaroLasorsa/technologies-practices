# Practice 010a: Terraform -- Basics & Local Resources

## Technologies

- **Terraform** -- Infrastructure as Code (IaC) tool by HashiCorp
- **HCL** -- HashiCorp Configuration Language (declarative DSL for infrastructure)
- **kreuzwerker/docker** -- Terraform provider for managing Docker resources locally
- **Docker Engine** -- Container runtime (already installed from previous practices)

## Stack

- HCL (Terraform configuration language)
- Docker provider (no cloud credentials needed)

## Theoretical Context

Terraform is an Infrastructure as Code (IaC) tool that enables declarative infrastructure management through configuration files written in HashiCorp Configuration Language (HCL). It solves the problem of manual infrastructure provisioning and configuration drift -- instead of clicking through cloud consoles or running ad-hoc scripts, you declare the desired state of your infrastructure in `.tf` files, and Terraform automatically figures out what actions (create, update, delete) are needed to reach that state. This declarative model enables version control, peer review, and reproducible environments.

Internally, Terraform's architecture consists of four core components: (1) **Terraform Core** -- a statically compiled binary (written in Go) that parses HCL configurations, builds a dependency graph of resources, and orchestrates the plan/apply workflow; (2) **Providers** -- plugins (also written in Go) that translate Terraform's generic resource definitions into API calls for specific platforms (AWS, Azure, Docker, Kubernetes, etc.); (3) **State file** -- a JSON file (`terraform.tfstate`) that stores the current state of managed infrastructure, mapping resource addresses (e.g., `docker_container.web`) to real-world identifiers (e.g., container ID `a3f8b92c...`); (4) **Backend** -- the storage mechanism for state (local filesystem, S3, Azure Blob, Terraform Cloud) that enables collaboration and state locking.

When you run `terraform apply`, the workflow is: (1) **Init** -- download provider plugins declared in `terraform { required_providers {...} }` and store them in `.terraform/` (happens once per project unless providers change); (2) **Parse** -- read all `.tf` files in the current directory, merge them into a single configuration tree; (3) **Refresh** -- query the real infrastructure via provider APIs to detect drift (manual changes made outside Terraform); (4) **Plan** -- compare desired state (HCL) with current state (JSON + API response) and compute the minimal set of actions (create, update, delete) needed; (5) **Apply** -- execute the plan by calling provider APIs in dependency order (Terraform's DAG ensures networks are created before containers attached to them); (6) **Update State** -- write the new resource IDs and attributes back to `terraform.tfstate` so future runs know what exists. The state file is **critical** -- losing it means Terraform forgets it ever created those resources and may try to create duplicates. Never edit it manually; use `terraform state` commands.

| Concept | Description |
|---------|-------------|
| **Resource** | A single infrastructure object (e.g., `docker_container.web`). Declared as `resource "<type>" "<name>" {...}`. |
| **Provider** | A plugin that communicates with an external API (e.g., `kreuzwerker/docker`). Downloaded during `terraform init`. |
| **State File** | JSON file tracking managed resources. Maps Terraform addresses to real IDs (container IDs, AWS ARNs, etc.). |
| **Plan** | Preview of actions Terraform will take. Shown as `+` (create), `~` (update), `-` (delete). Run `terraform plan` to see it. |
| **Apply** | Execute the plan. Prompts for confirmation unless `-auto-approve` is passed. Updates state after success. |
| **Destroy** | Delete all resources tracked in state. Runs a "delete" plan then applies it. Use `terraform destroy`. |
| **Variable** | Input parameter. Defined with `variable "name" {...}`, referenced with `var.name`. Overridable via CLI, tfvars, env. |
| **Output** | Computed value exposed after apply. Defined with `output "name" {...}`. Query with `terraform output`. |
| **Local** | Computed expression for DRY. Defined in `locals {...}`, referenced with `local.name`. Not overridable like variables. |
| **Data Source** | Query existing infrastructure without managing it. Declared as `data "<type>" "<name>" {...}`. Read-only. |

Terraform is the **most widely adopted IaC tool**, used across AWS, Azure, GCP, and on-prem infrastructure. Alternatives include **Pulumi** (supports real programming languages -- Python, TypeScript, Go -- instead of HCL, better for complex logic but smaller community), **CloudFormation** (AWS-only, YAML/JSON, tightly integrated but vendor-locked), **Ansible** (procedural, not declarative -- better for configuration management than infrastructure provisioning), and **CDK** (AWS CDK, Terraform CDK -- code-generates Terraform/CloudFormation, best of both worlds but adds complexity). Terraform's strengths are **provider ecosystem** (3000+ providers), **HCL readability** (more readable than YAML/JSON for infrastructure), and **state management** (explicit tracking of what exists). Its weaknesses are **state file brittleness** (corruption or loss can orphan resources), **HCL limitations** (not Turing-complete -- no complex loops, no functions), and **manual state resolution** (drift detection requires manual `terraform refresh` or `-refresh=true`).

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
