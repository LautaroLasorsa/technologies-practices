"""Practice 006a: Kubernetes Core Concepts -- Setup Guide.

Cross-platform setup instructions. Run: python setup.py
"""

import platform
import sys

STEPS = [
    (
        "Install minikube",
        {
            "Windows": [
                "winget install Kubernetes.minikube",
                "choco install minikube           (alternative)",
            ],
            "Darwin": ["brew install minikube"],
            "Linux": [
                "curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64",
                "sudo install minikube-linux-amd64 /usr/local/bin/minikube",
            ],
        },
        "Verify: minikube version",
    ),
    (
        "Install kubectl",
        {
            "Windows": [
                "winget install Kubernetes.kubectl",
                "choco install kubernetes-cli      (alternative)",
            ],
            "Darwin": ["brew install kubectl"],
            "Linux": ["sudo snap install kubectl --classic"],
        },
        "Verify: kubectl version --client",
    ),
    (
        "Start minikube cluster",
        {"all": ["minikube start --driver=docker"]},
        "Verify: kubectl cluster-info && kubectl get nodes",
    ),
    (
        "Point Docker CLI to minikube's daemon",
        {
            "Windows": [
                '(CMD)        @FOR /f "tokens=*" %i IN (\'minikube -p minikube docker-env --shell cmd\') DO @%i',
                "(PowerShell) & minikube -p minikube docker-env --shell powershell | Invoke-Expression",
                "(Git Bash)   eval $(minikube docker-env)",
            ],
            "Darwin": ["eval $(minikube docker-env)"],
            "Linux": ["eval $(minikube docker-env)"],
        },
        "This makes 'docker build' target minikube's internal Docker.",
    ),
    (
        "Build the app image",
        {"all": ["docker build -t task-tracker:v1 ./app"]},
        "Verify: docker images | grep task-tracker",
    ),
]


def main() -> None:
    os_name = platform.system()
    print(f"\n  Practice 006a: Kubernetes Core Concepts -- Setup")
    print(f"  Detected OS: {os_name}\n")

    for i, (title, commands, note) in enumerate(STEPS, 1):
        print(f"=== Step {i}: {title} ===\n")

        cmds = commands.get(os_name) or commands.get("all", [])
        for cmd in cmds:
            print(f"  {cmd}")

        print(f"\n  {note}\n")
        input("Press Enter to continue...")
        print()

    print("=== Ready! ===")
    print("Now open the k8s/ folder and start filling in the YAML manifests.\n")


if __name__ == "__main__":
    main()
