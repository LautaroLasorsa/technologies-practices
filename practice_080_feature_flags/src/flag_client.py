"""
OpenFeature + Flagsmith client setup -- shared across all services.

This module is the bridge between your application and the feature flag backend.
It encapsulates:
1. Flagsmith SDK initialization (connects to self-hosted Flagsmith API)
2. OpenFeature provider registration (vendor-agnostic abstraction)
3. Evaluation context construction (user attributes for targeting)

Architecture:
    Your code -> OpenFeature Client -> FlagsmithProvider -> Flagsmith API
                                                            |
                                                            v
                                                      PostgreSQL (flag storage)
"""

from __future__ import annotations

import os
import logging

from openfeature import api as openfeature_api
from openfeature.evaluation_context import EvaluationContext
from openfeature.client import OpenFeatureClient

from flagsmith import Flagsmith
from openfeature_flagsmith.provider import FlagsmithProvider

logger = logging.getLogger(__name__)

# Environment variables (set by docker-compose or .env)
FLAGSMITH_API_URL = os.environ.get("FLAGSMITH_API_URL", "http://localhost:8000/api/v1/")
FLAGSMITH_ENVIRONMENT_KEY = os.environ.get("FLAGSMITH_ENVIRONMENT_KEY", "")


def create_openfeature_client(domain: str = "default") -> OpenFeatureClient:
    """
    Initialize Flagsmith SDK, register an OpenFeature provider, and return a client.

    # -- Exercise Context ----------------------------------------------------------
    # This exercise teaches the core OpenFeature integration pattern. The OpenFeature
    # SDK provides a vendor-agnostic API: you register a "provider" (Flagsmith in this
    # case) once, then all flag evaluations go through the standard OpenFeature API.
    # This means if you later switch to LaunchDarkly or Unleash, you only change THIS
    # function -- all flag evaluation calls in your application remain unchanged.
    #
    # The Flagsmith Python SDK (flagsmith.Flagsmith) connects to the self-hosted
    # Flagsmith API. The FlagsmithProvider wraps it to implement the OpenFeature
    # provider interface. The OpenFeature API (openfeature.api) is the global
    # registry that maps domains to providers.
    # ------------------------------------------------------------------------------

    TODO(human): Implement this function.

    Steps:
    1. Create a Flagsmith client instance:
       - Use `Flagsmith(environment_key=..., api_url=...)` with the module-level
         constants FLAGSMITH_ENVIRONMENT_KEY and FLAGSMITH_API_URL.
       - The environment_key identifies which Flagsmith environment to read flags from.
       - The api_url points to your self-hosted Flagsmith instance.

    2. Create a FlagsmithProvider that wraps the Flagsmith client:
       - Use `FlagsmithProvider(client=flagsmith_client)`
       - This adapts Flagsmith's proprietary API to the OpenFeature interface.

    3. Register the provider with the OpenFeature API:
       - Use `openfeature_api.set_provider(provider, domain)` to register it
         under the given domain name.
       - Domains allow multiple providers in one app (e.g., different backends
         for different flag categories). For this practice, "default" suffices.

    4. Get and return an OpenFeature client:
       - Use `openfeature_api.get_client(domain)` to get a client bound to
         the registered provider.
       - This client is what your application uses for all flag evaluations.

    Returns:
        An OpenFeatureClient ready for flag evaluation.
    """
    raise NotImplementedError("TODO(human): Create Flagsmith client, provider, register, return OF client")


def build_evaluation_context(
    user_id: str,
    plan: str = "free",
    country: str = "US",
    beta_tester: bool = False,
    email: str = "",
) -> EvaluationContext:
    """
    Build an OpenFeature EvaluationContext from user attributes.

    # -- Exercise Context ----------------------------------------------------------
    # Evaluation context is how you tell the flag system "who" is requesting the
    # flag value. The flag backend uses this context for:
    # - Targeting rules: "enable for users where plan == enterprise"
    # - Percentage rollouts: hash(targeting_key) determines the bucket
    # - Segment matching: "beta_users" segment matches beta_tester == True
    #
    # The targeting_key is the primary identifier used for percentage bucketing.
    # It MUST be consistent for the same user across requests (usually user ID).
    # Additional attributes go in the `attributes` dict for rule matching.
    #
    # In Flagsmith specifically, the targeting_key maps to Flagsmith's "identity"
    # concept, and attributes map to "traits" on that identity.
    # ------------------------------------------------------------------------------

    TODO(human): Implement this function.

    Steps:
    1. Create and return an EvaluationContext with:
       - targeting_key = user_id  (this is the primary key for bucketing)
       - attributes = dict containing: plan, country, beta_tester, email
         (these are used for segment matching and targeting rules)

    Hint: EvaluationContext(targeting_key="...", attributes={...})

    The targeting_key is critical: it determines which percentage bucket the user
    falls into for rollout flags. Two requests with the same targeting_key MUST
    get the same flag value (sticky bucketing).

    Args:
        user_id: Unique user identifier (becomes the targeting key).
        plan: Subscription plan ("free", "starter", "enterprise").
        country: ISO country code.
        beta_tester: Whether the user is in the beta program.
        email: User email address.

    Returns:
        An EvaluationContext ready for flag evaluation.
    """
    raise NotImplementedError("TODO(human): Build EvaluationContext with targeting_key and attributes")
