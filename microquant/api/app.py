"""
FastAPI Application Factory for MicroQuant API Server.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from microquant.core.config import settings
from microquant.api.routes import router


def create_app() -> FastAPI:
    """Creates and configures the FastAPI application instance."""
    app = FastAPI(
        title="Aetheris-MicroQuant API",
        description="High-Frequency Order Book Microstructure Alpha Prediction and Backtesting Engine",
        version=settings.VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Configure CORS for dashboard and external quants
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")
    return app


app = create_app()
