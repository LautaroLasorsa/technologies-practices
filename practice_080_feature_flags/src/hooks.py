"""
OpenFeature Hooks -- lifecycle callbacks for flag evaluation.

Hooks let you add cross-cutting concerns to every flag evaluation without
modifying the evaluation call sites. They run at four stages:

    before -> [provider resolves flag] -> after -> finally_after
                      |
                      v (if error)
                    error -> finally_after

Common hook use cases:
- Logging: Record every flag evaluation for debugging and audit
- Metrics: Track evaluation count, latency, error rate
- Validation: Reject unexpected flag values before they reach application code
- Context enrichment: Add attributes to evaluation context in the `before` stage
- Telemetry: Push flag evaluation events to analytics (Segment, Amplitude)
"""

from __future__ import annotations

import logging
import time
from typing import Any

from openfeature.hook import Hook
from openfeature.flag_evaluation import FlagEvaluationDetails
from openfeature.hook import HookContext, HookHints

logger = logging.getLogger(__name__)


# =============================================================================
# In-memory metrics store (production would use Prometheus, Datadog, etc.)
# =============================================================================

class FlagMetrics:
    """Thread-safe-ish metrics store for flag evaluations.

    In production, you would push these to a metrics backend (Prometheus counters,
    Datadog gauges). For this practice, we accumulate in memory and expose via API.
    """

    def __init__(self) -> None:
        self.evaluation_count: int = 0
        self.error_count: int = 0
        self.total_latency_ms: float = 0.0
        self.per_flag_count: dict[str, int] = {}
        self.per_flag_variants: dict[str, dict[str, int]] = {}

    def record_evaluation(self, flag_key: str, variant: str | None, latency_ms: float) -> None:
        """Record a successful flag evaluation."""
        self.evaluation_count += 1
        self.total_latency_ms += latency_ms
        self.per_flag_count[flag_key] = self.per_flag_count.get(flag_key, 0) + 1
        if variant:
            if flag_key not in self.per_flag_variants:
                self.per_flag_variants[flag_key] = {}
            variants = self.per_flag_variants[flag_key]
            variants[variant] = variants.get(variant, 0) + 1

    def record_error(self, flag_key: str) -> None:
        """Record a flag evaluation error."""
        self.error_count += 1
        self.per_flag_count[flag_key] = self.per_flag_count.get(flag_key, 0) + 1

    def summary(self) -> dict[str, Any]:
        """Return a metrics summary dict."""
        avg_latency = (
            self.total_latency_ms / self.evaluation_count
            if self.evaluation_count > 0
            else 0.0
        )
        return {
            "total_evaluations": self.evaluation_count,
            "total_errors": self.error_count,
            "average_latency_ms": round(avg_latency, 2),
            "per_flag_count": dict(self.per_flag_count),
            "per_flag_variants": {k: dict(v) for k, v in self.per_flag_variants.items()},
        }


# Singleton metrics instance (shared across hooks and API)
metrics = FlagMetrics()


# =============================================================================
# Logging Hook
# =============================================================================

class LoggingHook(Hook):
    """
    Logs every flag evaluation with flag key, resolved value, and user context.

    # -- Exercise Context ----------------------------------------------------------
    # This hook demonstrates the `after` lifecycle stage. After the provider resolves
    # a flag value, the `after` method receives:
    # - hook_context: Contains flag_key, flag_type, default_value, and the
    #   evaluation_context (user attributes) that was used for resolution.
    # - details: Contains the resolved value, variant name, reason, and any
    #   error information.
    # - hints: Optional key-value pairs passed from the evaluation call site.
    #
    # In production, this hook is essential for debugging: "why did user X see
    # variant Y?" The log line answers this by recording the flag key, resolved
    # value, variant, and the targeting key (user ID) from the evaluation context.
    #
    # Hook methods return Optional[EvaluationContext] in the `before` stage (to
    # modify context) and None in all other stages.
    # ------------------------------------------------------------------------------

    TODO(human): Implement the `after` method.

    The `after` method receives:
    - hook_context (HookContext): has .flag_key, .flag_type, .default_value,
      and .evaluation_context (the EvaluationContext with targeting_key and attributes)
    - details (FlagEvaluationDetails): has .value (resolved value), .variant
      (variant name or None), .reason (why this value was chosen)
    - hints (HookHints): optional hints dict

    Steps:
    1. Extract the targeting_key from hook_context.evaluation_context
       (this is the user ID). Handle the case where evaluation_context is None.
    2. Log at INFO level a message like:
       "Flag '{flag_key}' evaluated to '{value}' (variant={variant}) for user '{targeting_key}'"
    3. Return None (after hooks don't modify anything).

    Hint: hook_context.evaluation_context.targeting_key gives you the user ID.
          details.value is the resolved flag value.
          details.variant is the variant name (may be None for simple flags).
    """

    def after(
        self,
        hook_context: HookContext,
        details: FlagEvaluationDetails,
        hints: HookHints,
    ) -> None:
        raise NotImplementedError("TODO(human): Log flag evaluation in after hook")


# =============================================================================
# Metrics Hook
# =============================================================================

class MetricsHook(Hook):
    """
    Tracks flag evaluation count, per-flag counts, variant distribution, and latency.

    # -- Exercise Context ----------------------------------------------------------
    # This hook demonstrates using BOTH the `before` and `after` stages together.
    # The `before` stage records the start time, and the `after` stage calculates
    # the elapsed time. This is the standard pattern for measuring operation latency
    # with hooks.
    #
    # The challenge: hooks are stateless between stages. You cannot store the start
    # time in an instance variable because multiple evaluations may overlap
    # (concurrent requests). The solution is to use the `hints` dict -- but since
    # hints are read-only in the OpenFeature spec, we use a thread-local or a
    # simpler approach: store the start time as an attribute on the hook_context
    # (which is created fresh for each evaluation).
    #
    # For this practice, we use a simpler approach: store start times in a dict
    # keyed by id(hook_context) on the hook instance. This works for single-threaded
    # async code (FastAPI with uvicorn).
    # ------------------------------------------------------------------------------

    TODO(human): Implement both `before` and `after` methods.

    The `before` method:
    - Receives hook_context and hints
    - Should record the current time (time.perf_counter()) in self._start_times
      keyed by id(hook_context). This is so the `after` method can calculate latency.
    - Returns None (we don't modify the evaluation context here)

    The `after` method:
    - Receives hook_context, details, and hints
    - Should calculate the elapsed time since `before` using self._start_times
    - Record the evaluation in the metrics singleton:
      metrics.record_evaluation(flag_key, variant, latency_ms)
    - Clean up the start time entry from self._start_times
    - Returns None

    The `error` method (bonus, simpler):
    - Receives hook_context, exception, and hints
    - Record the error: metrics.record_error(flag_key)
    - Clean up the start time entry
    - Returns None

    Hint: time.perf_counter() gives high-resolution timing.
          Latency in ms = (end - start) * 1000
          hook_context.flag_key gives the flag name.
          details.variant gives the variant (may be None).
    """

    def __init__(self) -> None:
        super().__init__()
        self._start_times: dict[int, float] = {}

    def before(
        self,
        hook_context: HookContext,
        hints: HookHints,
    ) -> None:
        raise NotImplementedError("TODO(human): Record start time in before hook")

    def after(
        self,
        hook_context: HookContext,
        details: FlagEvaluationDetails,
        hints: HookHints,
    ) -> None:
        raise NotImplementedError("TODO(human): Calculate latency and record metrics in after hook")

    def error(
        self,
        hook_context: HookContext,
        exception: Exception,
        hints: HookHints,
    ) -> None:
        raise NotImplementedError("TODO(human): Record error in metrics")
