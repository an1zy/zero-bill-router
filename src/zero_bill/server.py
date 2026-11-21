import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, StreamingResponse

from zero_bill.config import AppConfig
from zero_bill.models import ChatCompletionRequest
from zero_bill.router import (
    AllProvidersFailedError,
    NoActiveProvidersError,
    Router,
    RoutingError,
)

logger = logging.getLogger("zero_bill.server")


def create_app(
    config: AppConfig,
    tracker: Any | None = None,
    client: Any | None = None,
) -> FastAPI:
    """Create and configure FastAPI application for zero-bill-router."""
    router = Router(config=config, client=client, tracker=tracker)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        await router.close()

    app = FastAPI(
        title="zero-bill-router",
        description="Open-source CLI and local OpenAI-compatible proxy router",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Store references on app state for introspection/testing
    app.state.config = config
    app.state.router = router
    app.state.tracker = tracker

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Health check endpoint exposing service status and active providers."""
        active_providers = [p.name for p in router.get_active_providers()]
        return {
            "status": "ok",
            "ollama_enabled": config.ollama.enabled,
            "active_providers": active_providers,
        }

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        """OpenAI-compatible models listing."""
        created_timestamp = int(time.time())
        seen_ids: set[str] = set()
        model_data: list[dict[str, Any]] = []

        def register_model(model_id: str, owned_by: str) -> None:
            if model_id and model_id not in seen_ids:
                seen_ids.add(model_id)
                model_data.append(
                    {
                        "id": model_id,
                        "object": "model",
                        "created": created_timestamp,
                        "owned_by": owned_by,
                    }
                )

        # Primary auto routing alias
        register_model("zero-bill-auto", "zero-bill-router")

        # Local Ollama models
        if config.ollama.enabled:
            register_model("ollama", "ollama")
            register_model(config.ollama.model, "ollama")

        for provider in config.providers:
            register_model(provider.name, provider.name)
            register_model(provider.model, provider.name)

        return {
            "object": "list",
            "data": model_data,
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> Response:
        """OpenAI-compatible chat completions proxy endpoint."""
        try:
            if request.stream:
                stream_generator = await router.route_chat_completion_stream(request)
                return StreamingResponse(
                    stream_generator,
                    media_type="text/event-stream",
                )

            data = await router.route_chat_completion(request)
            headers: dict[str, str] = {}
            if isinstance(data, dict):
                provider_name = data.get("_provider") or data.get("provider")
                if provider_name:
                    headers["X-ZeroBill-Provider"] = str(provider_name)

            return JSONResponse(
                content=data,
                headers=headers if headers else None,
            )

        except (AllProvidersFailedError, NoActiveProvidersError, RoutingError) as exc:
            logger.error("Routing failed: %s", exc)
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": str(exc),
                        "type": "upstream_error",
                    }
                },
            )
        except Exception as exc:
            logger.exception("Unexpected error during chat completion: %s", exc)
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"Proxy routing failure: {str(exc)}",
                        "type": "upstream_error",
                    }
                },
            )

    return app
