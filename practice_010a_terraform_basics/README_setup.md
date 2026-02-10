# Setup: Terraform + Docker Provider on Windows

## Prerequisites

- **Docker Desktop** installed and running (you should already have this from previous practices)
- **Windows Terminal** or PowerShell

## Step 1: Install Terraform

### Option A: Chocolatey (recommended)

```powershell
choco install terraform
```

### Option B: Manual installation

1. Download the Windows AMD64 zip from: https://developer.hashicorp.com/terraform/install
2. Extract `terraform.exe` to a directory, e.g., `C:\terraform\`
3. Add the directory to your PATH:

```powershell
# PowerShell (user-level, persistent)
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\terraform", "User")
```

4. Restart your terminal.

### Verify

```powershell
terraform -version
# Expected: Terraform v1.x.x
```

## Step 2: Verify Docker is running

```powershell
docker info
# Should show Docker Engine details without errors
```

The Docker provider communicates with Docker via the Docker API. On Windows with Docker Desktop, this works automatically through the named pipe (`//./pipe/docker_engine`).

## Step 3: Run your first exercise

```powershell
cd practice_010a_terraform_basics\01_hello

# Download the Docker provider plugin
terraform init

# Preview what Terraform will create
terraform plan

# Create the resources (type "yes" when prompted)
terraform apply

# Verify: open http://localhost:8080 in browser

# Tear everything down
terraform destroy
```

## Useful commands

| Command | Description |
|---------|-------------|
| `terraform init` | Download provider plugins (run once per directory) |
| `terraform plan` | Preview changes without applying |
| `terraform apply` | Create/update resources |
| `terraform apply -auto-approve` | Apply without confirmation prompt |
| `terraform destroy` | Delete all managed resources |
| `terraform fmt` | Auto-format .tf files (like `rustfmt`) |
| `terraform validate` | Check syntax and internal consistency |
| `terraform output` | Show current output values |
| `terraform state list` | List all resources in state |
| `terraform state show <addr>` | Show details of a specific resource |

## References

- [Install Terraform](https://developer.hashicorp.com/terraform/install)
- [Terraform Docker Get Started](https://developer.hashicorp.com/terraform/tutorials/docker-get-started)
- [kreuzwerker/docker Provider](https://registry.terraform.io/providers/kreuzwerker/docker/latest/docs)
