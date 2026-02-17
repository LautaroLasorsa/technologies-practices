"""
Practice 058a — FastAPI Application Entry Point

This is the fully-implemented application entry point. It creates the FastAPI
app, registers the API router, and uses a lifespan handler to initialize the
database on startup.

The lifespan pattern (contextmanager) replaced the deprecated @app.on_event
startup/shutdown handlers in FastAPI 0.109+. Everything before `yield` runs
on startup, everything after runs on shutdown.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import router
from app.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler.

    - Startup: initialize DB engine and create tables.
    - Shutdown: (nothing to clean up — engine closes with the process).
    """
    # Import models so their tables are registered in SQLModel.metadata
    # BEFORE create_all() is called. Without this import, create_all()
    # wouldn't know about the Hero and Team tables.
    import app.models  # noqa: F401

    init_db()
    print("Database initialized. Tables created.")
    yield
    print("Application shutting down.")


def create_app() -> FastAPI:
    """Factory function to create and configure the FastAPI application."""
    application = FastAPI(
        title="SQLModel Heroes API",
        description="Practice 058a — SQLModel + FastAPI integration",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.include_router(router)
    return application


app = create_app()
