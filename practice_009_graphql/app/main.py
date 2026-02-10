"""FastAPI application entry point.

Mounts the Strawberry GraphQL router at /graphql with:
- GraphiQL IDE for interactive queries (GET /graphql)
- Query/mutation execution (POST /graphql)
- WebSocket subscriptions (WS /graphql)
- DataLoader context injection (per-request)
"""

from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter

from dataloader import get_context
from schema import schema

app = FastAPI(
    title="Bookstore GraphQL API",
    description="Practice 009: GraphQL with Strawberry + FastAPI",
)

graphql_router = GraphQLRouter(
    schema=schema,
    context_getter=get_context,
    graphql_ide="graphiql",
)

app.include_router(graphql_router, prefix="/graphql")
