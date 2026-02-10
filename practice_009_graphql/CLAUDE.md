# Practice 009: GraphQL

## Technologies

- **GraphQL** — Query language and runtime for APIs with strong typing and introspection
- **Strawberry** — Code-first Python GraphQL library using type hints
- **FastAPI** — ASGI web framework (Strawberry integration via `GraphQLRouter`)
- **DataLoader** — Batch/cache pattern to solve N+1 queries

## Stack

- Python 3.12+ (uv)
- Docker / Docker Compose

## Description

Build a **Bookstore API** that demonstrates core GraphQL patterns: queries, mutations, subscriptions, cursor-based pagination, DataLoader for N+1 prevention, and errors-as-data — all using Strawberry's code-first approach integrated with FastAPI.

### What you'll learn

1. **GraphQL fundamentals** — schema, types, queries, mutations, subscriptions, resolvers
2. **Code-first approach** — Python type hints generate SDL automatically (Strawberry vs schema-first)
3. **N+1 problem & DataLoader** — why independent resolvers cause cascading queries and how batching fixes it
4. **Cursor-based pagination** — Relay Connection spec (edges, pageInfo, opaque cursors)
5. **Error handling patterns** — errors-as-data vs exceptions, partial responses with nullable fields
6. **Real-time subscriptions** — async generators over WebSocket
7. **GraphQL vs REST** — when each shines, tradeoffs (caching, complexity, flexibility)

## Instructions

### Phase 1: Setup & Concepts (~10 min)

1. Set up Docker Compose with a single `api` service (Python + uvicorn)
2. Initialize Python project with `uv`, install `strawberry-graphql[fastapi]`
3. Understand the GraphQL execution model: parse → validate → resolve → format
4. Key question: What does "code-first" mean vs "schema-first"? What are the tradeoffs?

### Phase 2: Types & Basic Queries (~20 min)

1. Define `Book` and `Author` types with `@strawberry.type`
2. Create an in-memory data store (dicts/lists)
3. **User implements:** Query resolvers for `books` (list all) and `book(id)` (single lookup)
4. **User implements:** Nested resolver — `Book.author` that resolves the author for each book
5. Test via GraphiQL: query books with nested author fields
6. Key question: When you query 10 books with their authors, how many times does the author resolver run?

### Phase 3: Mutations & Error Handling (~15 min)

1. Define `BookInput` with `@strawberry.input`
2. Define `MutationResult` type with `success: bool` and `errors: list[str]`
3. **User implements:** `addBook` mutation with input validation (errors-as-data pattern)
4. **User implements:** `updateBook` mutation — handle "not found" gracefully
5. Key question: Why return errors in the response body instead of throwing exceptions?

### Phase 4: DataLoader (~20 min)

1. Observe the N+1 problem: add logging to see how many times author lookup is called
2. Understand DataLoader: collects `.load(key)` calls, executes one batch function
3. **User implements:** `AuthorLoader` batch function that resolves multiple author IDs in one call
4. Wire DataLoader into Strawberry's context (per-request instance)
5. Verify: same query now makes 1 batch call instead of N individual calls
6. Key question: Why must DataLoader instances be created per-request, not globally?

### Phase 5: Cursor-Based Pagination (~20 min)

1. Understand Relay Connection spec: `BookConnection`, `BookEdge`, `PageInfo`
2. Define connection types with `@strawberry.type`
3. **User implements:** `books` query with `first` and `after` arguments, returning paginated results
4. Implement opaque cursors (base64-encoded index or ID)
5. Test: paginate through books 3 at a time
6. Key question: Why is cursor-based pagination better than offset for dynamic data?

### Phase 6: Subscriptions (~25 min)

1. Set up WebSocket support in FastAPI for Strawberry subscriptions
2. **User implements:** `bookAdded` subscription using `async generator`
3. Use an `asyncio.Queue` or similar to broadcast new book events
4. Test: open GraphiQL subscription, add a book via mutation, see real-time update
5. Key question: What transport protocol do GraphQL subscriptions use? How does it differ from REST polling?

### Phase 7: Discussion (~10 min)

1. When would you choose GraphQL over REST? When would REST be better?
2. How would you handle authentication/authorization in GraphQL?
3. How would you prevent expensive/abusive queries (query depth limiting, complexity analysis)?

## Motivation

- **Modern API design**: GraphQL is widely adopted (GitHub, Shopify, Stripe) for flexible client-driven APIs
- **Strawberry + FastAPI**: Pythonic, type-safe stack gaining traction as the modern alternative to Graphene
- **N+1 awareness**: DataLoader is a universal pattern applicable beyond GraphQL (any batching scenario)
- **Full-stack literacy**: Understanding GraphQL tradeoffs vs REST is essential for architecture decisions
- **Real-time patterns**: Subscriptions introduce event-driven thinking applicable to WebSockets, SSE, and messaging

## References

- [GraphQL Official Docs](https://graphql.org/learn/)
- [Strawberry Documentation](https://strawberry.rocks/)
- [Strawberry FastAPI Integration](https://strawberry.rocks/docs/integrations/fastapi)
- [Strawberry DataLoaders](https://strawberry.rocks/docs/guides/dataloaders)
- [Strawberry Subscriptions](https://strawberry.rocks/docs/general/subscriptions)
- [GraphQL Pagination](https://graphql.org/learn/pagination/)
- [N+1 Problem Explained](https://hygraph.com/blog/graphql-n-1-problem)
- [GraphQL Error Handling](https://productionreadygraphql.com/2020-08-01-guide-to-graphql-errors/)
- [GraphQL vs REST](https://api7.ai/blog/graphql-vs-rest-api-comparison-2025)

## Commands

### Infrastructure (Docker)

| Command | Description |
|---------|-------------|
| `docker compose up -d --build` | Build the image and start the API container in detached mode |
| `docker compose up -d` | Start the API container (skip build if image exists) |
| `docker compose down` | Stop and remove the API container |
| `docker compose logs -f api` | Follow the API container logs (uvicorn output) |
| `docker compose build --no-cache` | Force a full image rebuild (e.g., after dependency changes) |

### Python environment (run from `app/`, for local development without Docker)

| Command | Description |
|---------|-------------|
| `uv sync` | Install Python dependencies from the lockfile |
| `uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload` | Run the API locally with auto-reload |

### Verification & Testing

| Command | Description |
|---------|-------------|
| Open `http://localhost:8000/graphql` | GraphiQL IDE for interactive queries, mutations, and subscriptions |

## State

`not-started`
