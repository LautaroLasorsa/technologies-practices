"""gRPC server interceptors.

Interceptors are the gRPC equivalent of HTTP middleware. They wrap every RPC
call, letting you add cross-cutting concerns (logging, auth, metrics) without
modifying individual service methods.

A server interceptor implements `grpc.ServerInterceptor` and its single method
`intercept_service(continuation, handler_call_details)`:
  - `continuation` is a callable that invokes the next interceptor (or the
    actual handler if this is the last one). Call it to proceed.
  - `handler_call_details` carries metadata about the incoming RPC: method
    name, invocation metadata, etc.
  - Return value: the RPC handler (what continuation returns), or a custom
    handler to short-circuit (e.g., for auth rejection).

Docs: https://grpc.io/docs/guides/interceptors/
API:  https://grpc.github.io/grpc/python/grpc.html#grpc.ServerInterceptor
"""

from __future__ import annotations

import grpc


class LoggingInterceptor(grpc.ServerInterceptor):
    """Logs every RPC call: method name, start time, and duration.

    Example output:
        [gRPC] /taskmanager.TaskManager/CreateTask | 2024-01-15T10:30:00 | 3.2ms
    """

    def intercept_service(
        self,
        continuation: grpc.HandlerCallDetails,
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler | None:
        # ── Exercise Context ──────────────────────────────────────────────────
        # This exercise teaches gRPC interceptors: middleware that wraps every RPC call
        # for cross-cutting concerns (logging, metrics, auth). This is the pattern for
        # adding observability to services without modifying individual handlers.

        # TODO(human): Implement the logging interceptor.
        #
        # Steps:
        #   1. Extract the method name from `handler_call_details.method`
        #   2. Record the current time (datetime.now or time.perf_counter)
        #   3. Call `continuation(handler_call_details)` to get the actual handler
        #   4. Print/log the method name and timestamp
        #
        # Hint: The interceptor fires BEFORE the handler runs, so you can't
        # measure duration here directly. For duration, you'd need to wrap the
        # handler itself. For this exercise, just log the method name + timestamp
        # when the call arrives. That alone is useful for debugging.
        #
        # Advanced (optional): Wrap the returned handler to measure duration.
        # See: https://github.com/grpc/grpc/tree/master/examples/python/interceptors
        #
        # Signature reminder:
        #   continuation(handler_call_details) -> grpc.RpcMethodHandler | None
        raise NotImplementedError("TODO(human): implement intercept_service")
