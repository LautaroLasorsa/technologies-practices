# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.28"]
# ///
"""
End-to-end test for Feature Flags practice.

Tests all flag patterns against the running Product API and Recommendation Service.

Usage:
    uv run scripts/test_flags.py

Requires:
    - Flagsmith running and seeded (docker compose up + seed_flagsmith.py)
    - Product API on port 8001
    - Recommendation Service on port 8002
"""

from __future__ import annotations

import sys
import json

import httpx

PRODUCT_API = "http://localhost:8001"
RECO_SERVICE = "http://localhost:8002"

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"


def test_health_checks(client: httpx.Client) -> bool:
    """Test that both services are healthy."""
    print("\n--- Health Checks ---")
    ok = True

    for name, url in [("Product API", PRODUCT_API), ("Recommendation Service", RECO_SERVICE)]:
        try:
            resp = client.get(f"{url}/health")
            if resp.status_code == 200:
                print(f"  {PASS} {name} is healthy")
            else:
                print(f"  {FAIL} {name} returned {resp.status_code}")
                ok = False
        except httpx.ConnectError:
            print(f"  {FAIL} {name} not reachable at {url}")
            ok = False

    return ok


def test_products_endpoint(client: httpx.Client) -> None:
    """Test the /products endpoint (boolean flags: maintenance_mode + new_search_enabled)."""
    print("\n--- Products Endpoint (Boolean Flags) ---")

    # Test with a user
    headers = {"X-User-Id": "test-user-1", "X-User-Plan": "starter"}
    resp = client.get(f"{PRODUCT_API}/products", headers=headers)

    if resp.status_code == 503:
        print(f"  {PASS} Maintenance mode is ON (503 returned)")
        print(f"       Toggle 'maintenance_mode' OFF in Flagsmith to continue")
        return

    if resp.status_code == 200:
        data = resp.json()
        engine = data.get("search_engine", "unknown")
        count = len(data.get("products", []))
        print(f"  {PASS} Products returned: {count} items, search engine: {engine}")
    else:
        print(f"  {FAIL} Unexpected status: {resp.status_code} - {resp.text[:200]}")

    # Test with search query
    resp = client.get(f"{PRODUCT_API}/products?q=electronics", headers=headers)
    if resp.status_code == 200:
        data = resp.json()
        count = len(data.get("products", []))
        engine = data.get("search_engine", "unknown")
        print(f"  {PASS} Search 'electronics': {count} results (engine: {engine})")
        if engine == "new":
            print(f"       new_search_enabled flag is ON (category search works)")
        else:
            print(f"       new_search_enabled flag is OFF (name-only search)")


def test_algorithm_endpoint(client: httpx.Client) -> None:
    """Test the /algorithm endpoint (multivariate string flag)."""
    print("\n--- Algorithm Endpoint (Multivariate Flag) ---")

    headers = {"X-User-Id": "test-user-1"}
    resp = client.get(f"{PRODUCT_API}/algorithm", headers=headers)

    if resp.status_code == 200:
        data = resp.json()
        algo = data.get("algorithm", "unknown")
        print(f"  {PASS} Algorithm: {algo}")
        if algo in ("collaborative", "content_based", "hybrid"):
            print(f"       Valid algorithm value")
        else:
            print(f"  {FAIL} Unexpected algorithm: {algo}")
    else:
        print(f"  {FAIL} Status: {resp.status_code} - {resp.text[:200]}")


def test_ui_config_endpoint(client: httpx.Client) -> None:
    """Test the /ui-config endpoint (JSON remote config flag)."""
    print("\n--- UI Config Endpoint (JSON Remote Config) ---")

    headers = {"X-User-Id": "test-user-1"}
    resp = client.get(f"{PRODUCT_API}/ui-config", headers=headers)

    if resp.status_code == 200:
        data = resp.json()
        config = data.get("ui_config", {})
        print(f"  {PASS} UI Config: {json.dumps(config, indent=2)}")
        if "theme" in config:
            print(f"       Theme: {config['theme']}")
    else:
        print(f"  {FAIL} Status: {resp.status_code} - {resp.text[:200]}")


def test_checkout_experiment(client: httpx.Client) -> None:
    """Test the /checkout endpoint (percentage-based A/B experiment)."""
    print("\n--- Checkout Experiment (A/B Percentage Rollout) ---")

    # Test with multiple users to see variant distribution
    variants: dict[str, int] = {}
    for i in range(10):
        headers = {"X-User-Id": f"experiment-user-{i}"}
        resp = client.get(f"{PRODUCT_API}/checkout", headers=headers)
        if resp.status_code == 200:
            variant = resp.json().get("variant", "unknown")
            variants[variant] = variants.get(variant, 0) + 1

    if variants:
        print(f"  {PASS} Variant distribution across 10 users:")
        for variant, count in sorted(variants.items()):
            print(f"       {variant}: {count} users")

    # Test sticky bucketing: same user should get same variant
    headers = {"X-User-Id": "sticky-test-user"}
    variant1 = client.get(f"{PRODUCT_API}/checkout", headers=headers).json().get("variant")
    variant2 = client.get(f"{PRODUCT_API}/checkout", headers=headers).json().get("variant")
    if variant1 == variant2:
        print(f"  {PASS} Sticky bucketing works: same user gets '{variant1}' both times")
    else:
        print(f"  {FAIL} Sticky bucketing broken: got '{variant1}' then '{variant2}'")


def test_pricing_tiers(client: httpx.Client) -> None:
    """Test the /pricing endpoint (user-targeted flag based on plan)."""
    print("\n--- Pricing Tiers (User-Targeted Flag) ---")

    for plan in ("free", "starter", "enterprise"):
        headers = {"X-User-Id": f"user-{plan}", "X-User-Plan": plan}
        resp = client.get(f"{PRODUCT_API}/pricing", headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            tier = data.get("tier", "unknown")
            features = data.get("features", {})
            print(f"  {PASS} Plan '{plan}' -> tier '{tier}', rate_limit={features.get('api_rate_limit', '?')}")
        else:
            print(f"  {FAIL} Plan '{plan}': {resp.status_code} - {resp.text[:200]}")


def test_cross_service_propagation(client: httpx.Client) -> None:
    """Test the /recommendations endpoint (cross-service flag propagation)."""
    print("\n--- Cross-Service Flag Propagation ---")

    headers = {
        "X-User-Id": "propagation-test-user",
        "X-User-Plan": "enterprise",
        "X-User-Country": "DE",
    }

    # Test via Product API proxy
    resp = client.get(f"{PRODUCT_API}/recommendations", headers=headers)
    if resp.status_code == 200:
        data = resp.json()
        source = data.get("source", "unknown")
        downstream = data.get("downstream_response", {})
        if downstream:
            algo = downstream.get("algorithm", "unknown")
            reco_count = len(downstream.get("recommendations", []))
            print(f"  {PASS} Product API proxied to Recommendation Service")
            print(f"       Source: {source}")
            print(f"       Algorithm: {algo}, Recommendations: {reco_count}")
            if downstream.get("user_id") == "propagation-test-user":
                print(f"  {PASS} User context propagated correctly")
            else:
                print(f"  {FAIL} User context not propagated (got: {downstream.get('user_id')})")
        else:
            # Might be a fallback response
            print(f"  {PASS} Got response (source: {source})")
            if "error" in data:
                print(f"       Fallback triggered: {data['error']}")
    else:
        print(f"  {FAIL} Status: {resp.status_code} - {resp.text[:200]}")

    # Test direct call to Recommendation Service
    resp = client.get(f"{RECO_SERVICE}/recommendations", headers=headers)
    if resp.status_code == 200:
        data = resp.json()
        print(f"  {PASS} Direct Recommendation Service call works")
        print(f"       Algorithm: {data.get('algorithm')}, User: {data.get('user_id')}")
    else:
        print(f"  {FAIL} Direct call: {resp.status_code} - {resp.text[:200]}")


def test_metrics(client: httpx.Client) -> None:
    """Test that flag evaluation metrics are being collected."""
    print("\n--- Flag Evaluation Metrics ---")

    for name, url in [("Product API", PRODUCT_API), ("Recommendation Service", RECO_SERVICE)]:
        resp = client.get(f"{url}/metrics")
        if resp.status_code == 200:
            data = resp.json()
            total = data.get("total_evaluations", 0)
            errors = data.get("total_errors", 0)
            avg_latency = data.get("average_latency_ms", 0)
            print(f"  {PASS} {name}: {total} evaluations, {errors} errors, avg latency: {avg_latency}ms")
            if data.get("per_flag_count"):
                for flag, count in data["per_flag_count"].items():
                    print(f"       {flag}: {count} evaluations")
        else:
            print(f"  {SKIP} {name} metrics not available ({resp.status_code})")


def main() -> None:
    print("=" * 60)
    print("Feature Flags End-to-End Tests")
    print("=" * 60)

    with httpx.Client(timeout=10.0) as client:
        if not test_health_checks(client):
            print("\nServices not ready. Start them first:")
            print("  uv run src/product_api.py")
            print("  uv run src/recommendation_service.py")
            sys.exit(1)

        test_products_endpoint(client)
        test_algorithm_endpoint(client)
        test_ui_config_endpoint(client)
        test_checkout_experiment(client)
        test_pricing_tiers(client)
        test_cross_service_propagation(client)
        test_metrics(client)

    print("\n" + "=" * 60)
    print("Tests complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Open Flagsmith dashboard: http://localhost:8000")
    print("  2. Toggle flags and re-run: uv run scripts/test_flags.py")
    print("  3. Try: enable 'maintenance_mode' and re-run tests")
    print("  4. Try: change 'recommendation_algorithm' to 'hybrid' and re-run")


if __name__ == "__main__":
    main()
