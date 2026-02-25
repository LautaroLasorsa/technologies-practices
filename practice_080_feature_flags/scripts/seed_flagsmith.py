# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.28"]
# ///
"""
Seed Flagsmith with project, environment, features, and segments.

Run this AFTER:
1. Flagsmith is running (docker compose up -d flagsmith)
2. You've created an admin account at http://localhost:8000/signup

This script uses the Flagsmith Admin API to programmatically create:
- A project ("Feature Flags Practice")
- Features with default values:
  - maintenance_mode (boolean, default: disabled)
  - new_search_enabled (boolean, default: disabled)
  - recommendation_algorithm (string, default: "collaborative")
  - ui_config (string/JSON, default: {"theme": "light", ...})
  - checkout_experiment (string, default: "control")
  - premium_features (string, default: "basic")
  - show_sponsored (boolean, default: disabled)

Usage:
    uv run scripts/seed_flagsmith.py

The script will prompt for your admin email and password if not set in
environment variables (FLAGSMITH_ADMIN_EMAIL, FLAGSMITH_ADMIN_PASSWORD).
"""

from __future__ import annotations

import os
import sys
import json
import getpass
import time

import httpx

FLAGSMITH_URL = os.environ.get("FLAGSMITH_URL", "http://localhost:8000")
API_URL = f"{FLAGSMITH_URL}/api/v1"

# Admin credentials (set via env vars or prompted)
ADMIN_EMAIL = os.environ.get("FLAGSMITH_ADMIN_EMAIL", "")
ADMIN_PASSWORD = os.environ.get("FLAGSMITH_ADMIN_PASSWORD", "")


def get_credentials() -> tuple[str, str]:
    """Get admin email and password from env vars or interactive prompt."""
    email = ADMIN_EMAIL or input("Flagsmith admin email: ")
    password = ADMIN_PASSWORD or getpass.getpass("Flagsmith admin password: ")
    return email, password


def authenticate(client: httpx.Client, email: str, password: str) -> str:
    """Authenticate and return the API token."""
    resp = client.post(
        f"{API_URL}/auth/login/",
        json={"email": email, "password": password},
    )
    if resp.status_code != 200:
        print(f"Authentication failed (HTTP {resp.status_code}): {resp.text}")
        print("Make sure you've created an account at http://localhost:8000/signup")
        sys.exit(1)
    token = resp.json()["key"]
    print(f"  Authenticated as {email}")
    return token


def get_or_create_organisation(client: httpx.Client) -> int:
    """Get the first organisation (created with the admin account)."""
    resp = client.get(f"{API_URL}/organisations/")
    resp.raise_for_status()
    orgs = resp.json()["results"]
    if not orgs:
        print("No organisation found. Create one in the Flagsmith dashboard first.")
        sys.exit(1)
    org_id = orgs[0]["id"]
    print(f"  Organisation: {orgs[0]['name']} (id={org_id})")
    return org_id


def get_or_create_project(client: httpx.Client, org_id: int) -> int:
    """Get or create the practice project."""
    project_name = "Feature Flags Practice"
    resp = client.get(f"{API_URL}/projects/")
    resp.raise_for_status()
    for proj in resp.json():
        if proj["name"] == project_name:
            print(f"  Project already exists: {project_name} (id={proj['id']})")
            return proj["id"]

    resp = client.post(
        f"{API_URL}/projects/",
        json={"name": project_name, "organisation": org_id},
    )
    resp.raise_for_status()
    proj_id = resp.json()["id"]
    print(f"  Created project: {project_name} (id={proj_id})")
    return proj_id


def get_environment(client: httpx.Client, project_id: int) -> tuple[int, str]:
    """Get the Development environment and its API key."""
    resp = client.get(f"{API_URL}/environments/?project={project_id}")
    resp.raise_for_status()
    for env in resp.json():
        if env["name"] == "Development":
            env_id = env["id"]
            api_key = env["api_key"]
            print(f"  Environment: Development (id={env_id})")
            return env_id, api_key

    print("No 'Development' environment found (should be auto-created with project)")
    sys.exit(1)


def create_feature(
    client: httpx.Client,
    project_id: int,
    name: str,
    *,
    default_enabled: bool = False,
    initial_value: str = "",
    description: str = "",
) -> int | None:
    """Create a feature flag if it doesn't exist."""
    # Check if feature already exists
    resp = client.get(f"{API_URL}/projects/{project_id}/features/")
    resp.raise_for_status()
    for feat in resp.json():
        if feat["name"] == name:
            print(f"    Feature already exists: {name} (id={feat['id']})")
            return feat["id"]

    resp = client.post(
        f"{API_URL}/projects/{project_id}/features/",
        json={
            "name": name,
            "default_enabled": default_enabled,
            "initial_value": initial_value,
            "description": description,
        },
    )
    if resp.status_code in (200, 201):
        feat_id = resp.json()["id"]
        print(f"    Created feature: {name} (id={feat_id}, enabled={default_enabled})")
        return feat_id
    else:
        print(f"    Failed to create feature {name}: {resp.status_code} {resp.text}")
        return None


def seed_features(client: httpx.Client, project_id: int) -> None:
    """Create all practice feature flags."""
    print("\n  Creating features...")

    features = [
        {
            "name": "maintenance_mode",
            "default_enabled": False,
            "initial_value": "",
            "description": "Kill switch: returns 503 when enabled. Toggle in dashboard to test.",
        },
        {
            "name": "new_search_enabled",
            "default_enabled": False,
            "initial_value": "",
            "description": "Release toggle: enables new search algorithm (name + category matching).",
        },
        {
            "name": "recommendation_algorithm",
            "default_enabled": True,
            "initial_value": "collaborative",
            "description": "Multivariate: which recommendation algorithm to use (collaborative|content_based|hybrid).",
        },
        {
            "name": "ui_config",
            "default_enabled": True,
            "initial_value": json.dumps({
                "theme": "light",
                "max_results": 10,
                "show_banner": False,
            }),
            "description": "JSON remote config: UI layout and theme configuration.",
        },
        {
            "name": "checkout_experiment",
            "default_enabled": True,
            "initial_value": "control",
            "description": "A/B experiment: checkout flow variant (control|streamlined|one_click).",
        },
        {
            "name": "premium_features",
            "default_enabled": True,
            "initial_value": "basic",
            "description": "User-targeted: feature tier based on plan (basic|standard|premium).",
        },
        {
            "name": "show_sponsored",
            "default_enabled": False,
            "initial_value": "",
            "description": "Boolean: show sponsored recommendations in the recommendation service.",
        },
    ]

    for feat_def in features:
        create_feature(client, project_id, **feat_def)


def main() -> None:
    print("=" * 60)
    print("Flagsmith Seed Script")
    print("=" * 60)

    # Wait for Flagsmith to be ready
    print("\nChecking Flagsmith availability...")
    for attempt in range(30):
        try:
            resp = httpx.get(f"{FLAGSMITH_URL}/health", timeout=3.0)
            if resp.status_code == 200:
                print("  Flagsmith is ready!")
                break
        except httpx.ConnectError:
            pass
        if attempt == 29:
            print("  Flagsmith not responding after 30 attempts. Is it running?")
            sys.exit(1)
        time.sleep(2)

    email, password = get_credentials()

    with httpx.Client(timeout=10.0) as client:
        # Authenticate
        print("\nAuthenticating...")
        token = authenticate(client, email, password)
        client.headers["Authorization"] = f"Token {token}"

        # Get organisation
        print("\nGetting organisation...")
        org_id = get_or_create_organisation(client)

        # Create project
        print("\nSetting up project...")
        project_id = get_or_create_project(client, org_id)

        # Get environment
        print("\nGetting environment...")
        env_id, api_key = get_environment(client, project_id)

        # Create features
        seed_features(client, project_id)

    # Output the environment key
    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print(f"\nEnvironment API Key: {api_key}")
    print(f"\nSet this in your environment before running the services:")
    print(f"  export FLAGSMITH_ENVIRONMENT_KEY='{api_key}'")
    print(f"\nOr add to a .env file in the practice directory:")
    print(f"  FLAGSMITH_ENVIRONMENT_KEY={api_key}")
    print(f"\nFlagsmith Dashboard: {FLAGSMITH_URL}")
    print(f"  - Toggle flags in: Features tab")
    print(f"  - View identities: Identities tab")
    print(f"  - Manage segments: Segments tab")

    # Write .env file for convenience
    env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    with open(env_file, "w") as f:
        f.write(f"FLAGSMITH_ENVIRONMENT_KEY={api_key}\n")
    print(f"\n  Wrote {env_file} with FLAGSMITH_ENVIRONMENT_KEY")


if __name__ == "__main__":
    main()
