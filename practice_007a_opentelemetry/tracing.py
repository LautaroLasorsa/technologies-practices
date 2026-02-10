"""OpenTelemetry tracing setup.

Configures the global TracerProvider to export spans to Jaeger via OTLP/gRPC.

Run Jaeger first:
    docker compose up -d

Concepts:
    - Resource: identifies *this* service (name, version, environment)
    - TracerProvider: the central object that creates Tracers
    - BatchSpanProcessor: batches finished spans and exports them periodically
      (more efficient than exporting one-by-one)
    - OTLPSpanExporter: sends spans to any OTLP-compatible backend (Jaeger, Tempo, etc.)

Docs: https://opentelemetry.io/docs/languages/python/instrumentation/#tracing
"""

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# ── Configuration ────────────────────────────────────────────────────

JAEGER_OTLP_ENDPOINT = "http://localhost:4317"


# ── TODO(human): Implement these two functions ───────────────────────


def init_tracer(service_name: str, service_version: str = "0.1.0") -> None:
    """Initialize the global TracerProvider and attach an OTLP exporter.

    TODO(human): Implement this function.

    Steps:
      1. Create a Resource that identifies this service. Use Resource.create()
         with a dict containing:
             "service.name": service_name
             "service.version": service_version
         Resource is metadata attached to every span this service produces.
         Think of it as the "return address" on each trace.

      2. Create a TracerProvider, passing the resource:
             provider = TracerProvider(resource=resource)

      3. Create an OTLPSpanExporter pointing to Jaeger:
             exporter = OTLPSpanExporter(endpoint=JAEGER_OTLP_ENDPOINT, insecure=True)
         `insecure=True` disables TLS (fine for local Jaeger, never in production).

      4. Create a BatchSpanProcessor wrapping the exporter:
             processor = BatchSpanProcessor(exporter)
         The BatchSpanProcessor collects finished spans in memory and exports
         them in batches (default: every 5 seconds or 512 spans). This is more
         efficient than exporting each span immediately (SimpleSpanProcessor).

      5. Add the processor to the provider:
             provider.add_span_processor(processor)

      6. Set this provider as the global tracer provider:
             trace.set_tracer_provider(provider)

    After this function runs, any call to trace.get_tracer() anywhere in the
    process will use this provider (and thus export to Jaeger).

    Docs: https://opentelemetry.io/docs/languages/python/instrumentation/#initialize-tracing
    """
    raise NotImplementedError("TODO(human): implement init_tracer")


def get_tracer(name: str) -> trace.Tracer:
    """Return a named Tracer from the global TracerProvider.

    TODO(human): Implement this function.

    Steps:
      1. Call trace.get_tracer(name) and return the result.

    A Tracer is a lightweight handle used to create spans. The `name` parameter
    is typically the module name (e.g., "order_api") — it appears in Jaeger as
    the "instrumentation library" and helps you identify which code produced
    each span.

    Docs: https://opentelemetry.io/docs/languages/python/instrumentation/#acquiring-a-tracer
    """
    raise NotImplementedError("TODO(human): implement get_tracer")
