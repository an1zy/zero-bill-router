import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator
import httpx

from zero_bill.config import AppConfig, ProviderConfig
from zero_bill.models import ChatCompletionRequest

logger = logging.getLogger("zero_bill.router")


class RoutingError(Exception):
    """Base exception for routing errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class AllProvidersFailedError(RoutingError):
    """Raised when all candidate providers (Ollama and cloud providers) fail."""

    def __init__(self, errors: dict[str, str]) -> None:
        self.errors = errors
        formatted = "; ".join(f"{name}: {err}" for name, err in errors.items())
        super().__init__(f"All candidate providers failed: {formatted}")


class NoActiveProvidersError(RoutingError):
    """Raised when no active providers are available and Ollama is disabled."""

    def __init__(self, message: str = "No active providers available") -> None:
        super().__init__(message)


def _build_endpoint_url(base_url: str, endpoint: str = "/chat/completions") -> str:
    cleaned = base_url.rstrip("/")
    if cleaned.endswith(endpoint):
        return cleaned
    return f"{cleaned}{endpoint}"


class Router:
    """Intelligent proxy router between local Ollama and free-tier cloud providers.

    Routes requests based on length, keywords, provider priorities, and cooldown states.
    """

    def __init__(
        self,
        config: AppConfig,
        client: httpx.AsyncClient | None = None,
        tracker: Any | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.config = config
        self.tracker: Any | None = tracker
        self.timeout = timeout
        self._external_client = client is not None
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout))
        self._cooldowns: dict[str, float] = {}

    async def close(self) -> None:
        """Close the underlying HTTP client if created internally."""
        if not self._external_client and not self.client.is_closed:
            await self.client.aclose()

    async def __aenter__(self) -> "Router":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    def is_in_cooldown(self, provider_name: str) -> bool:
        """Check if a provider is currently in cooldown period."""
        expiry = self._cooldowns.get(provider_name, 0.0)
        return time.time() < expiry

    def get_cooldown_remaining(self, provider_name: str) -> float:
        """Return remaining cooldown seconds for a provider."""
        expiry = self._cooldowns.get(provider_name, 0.0)
        return max(0.0, expiry - time.time())

    def mark_cooldown(self, provider_name: str, duration: int | None = None) -> None:
        """Put a provider into cooldown for a specified duration or its default setting."""
        if duration is None:
            matching = next(
                (p for p in self.config.providers if p.name == provider_name),
                None,
            )
            duration = matching.cooldown_seconds if matching else 60
        self._cooldowns[provider_name] = time.time() + duration
        logger.warning(
            "Provider '%s' marked in cooldown for %d seconds",
            provider_name,
            duration,
        )

    def clear_cooldown(self, provider_name: str) -> None:
        """Remove a provider from cooldown."""
        self._cooldowns.pop(provider_name, None)

    def clear_all_cooldowns(self) -> None:
        """Reset all active cooldowns."""
        self._cooldowns.clear()

    def get_active_providers(self) -> list[ProviderConfig]:
        """Return all active providers with non-empty API keys, sorted by priority."""
        active = [
            p
            for p in self.config.providers
            if p.active and p.is_active and bool(p.api_key)
        ]
        return sorted(active, key=lambda p: p.priority)

    def get_available_providers(self) -> list[ProviderConfig]:
        """Return active providers that are not currently in cooldown."""
        return [p for p in self.get_active_providers() if not self.is_in_cooldown(p.name)]

    def should_route_to_local(self, request: ChatCompletionRequest) -> bool:
        """Check whether the request qualifies for local Ollama processing.

        Matches if Ollama is enabled AND:
        - prompt text contains any keyword from routing.local_keywords, OR
        - total characters count is within routing.local_max_chars.
        """
        if not self.config.ollama.enabled:
            return False

        prompt_text = request.get_full_prompt_text().lower()
        total_chars = request.get_total_chars()

        for kw in self.config.routing.local_keywords:
            if kw.lower() in prompt_text:
                return True

        if total_chars <= self.config.routing.local_max_chars:
            return True

        return False

    def build_candidates_plan(
        self,
        request: ChatCompletionRequest,
    ) -> list[dict[str, Any]]:
        """Construct an ordered list of candidate execution targets for a request."""
        candidates: list[dict[str, Any]] = []
        prefers_local = self.should_route_to_local(request)

        if prefers_local and self.config.ollama.enabled:
            candidates.append(
                {
                    "name": "ollama",
                    "target_type": "ollama",
                    "base_url": self.config.ollama.base_url,
                    "model": self.config.ollama.model,
                    "api_key": None,
                    "provider": None,
                }
            )

        available_cloud = self.get_available_providers()
        if not available_cloud:
            active_cloud = self.get_active_providers()
            available_cloud = sorted(
                active_cloud,
                key=lambda p: self.get_cooldown_remaining(p.name),
            )

        for provider in available_cloud:
            candidates.append(
                {
                    "name": provider.name,
                    "target_type": "provider",
                    "base_url": provider.base_url,
                    "model": provider.model,
                    "api_key": provider.api_key,
                    "provider": provider,
                }
            )

        if not prefers_local and self.config.ollama.enabled:
            candidates.append(
                {
                    "name": "ollama",
                    "target_type": "ollama",
                    "base_url": self.config.ollama.base_url,
                    "model": self.config.ollama.model,
                    "api_key": None,
                    "provider": None,
                }
            )

        return candidates

    def _prepare_headers_and_payload(
        self,
        candidate: dict[str, Any],
        request: ChatCompletionRequest,
        stream: bool,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        url = _build_endpoint_url(candidate["base_url"], "/chat/completions")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json" if not stream else "text/event-stream",
        }

        api_key = candidate.get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        if candidate["name"] == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/zero-bill-router"
            headers["X-Title"] = "zero-bill-router"

        payload = request.model_dump(exclude_none=True)
        payload["model"] = candidate["model"]
        payload["stream"] = stream

        return url, headers, payload

    def _apply_candidate_failure(self, candidate: dict[str, Any]) -> None:
        if candidate["target_type"] != "provider":
            return
        prov = candidate.get("provider")
        duration = prov.cooldown_seconds if prov else 60
        self.mark_cooldown(candidate["name"], duration)

    def _record_tracker_usage(
        self,
        candidate: dict[str, Any],
        data: Any,
        request: ChatCompletionRequest,
    ) -> None:
        if self.tracker is None:
            return

        try:
            p_tokens, c_tokens = _extract_tokens_usage(data, request)
            is_local = candidate["target_type"] == "ollama"
            route_reason = "local" if is_local else "cloud"

            prov_obj = candidate.get("provider")
            cost_m = prov_obj.cost_per_1m_tokens if (prov_obj and not is_local) else 0.0
            actual_cost = ((p_tokens + c_tokens) * cost_m) / 1_000_000.0

            if hasattr(self.tracker, "record"):
                self.tracker.record(
                    provider=candidate["name"],
                    model=candidate["model"],
                    route_reason=route_reason,
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    actual_cost_usd=float(actual_cost),
                )
            elif hasattr(self.tracker, "record_usage"):
                self.tracker.record_usage(
                    provider=candidate["name"],
                    model=candidate["model"],
                    response=data,
                )
        except Exception as tracker_err:
            logger.warning("Tracker record error: %s", tracker_err)

    async def _try_candidate(
        self,
        candidate: dict[str, Any],
        request: ChatCompletionRequest,
    ) -> tuple[dict[str, Any] | None, str | None]:
        name = candidate["name"]
        url, headers, payload = self._prepare_headers_and_payload(
            candidate,
            request,
            stream=False,
        )

        try:
            response = await self.client.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            self._apply_candidate_failure(candidate)
            err_text = f"Connection error: {str(exc)}"
            logger.warning("Target '%s' request exception: %s", name, err_text)
            return None, err_text

        if response.status_code >= 400:
            self._apply_candidate_failure(candidate)
            err_msg = f"HTTP {response.status_code}: {response.text[:300]}"
            logger.warning("Target '%s' failed: %s", name, err_msg)
            return None, err_msg

        data = response.json()
        if isinstance(data, dict):
            data["_provider"] = name

        self._record_tracker_usage(candidate, data, request)
        return data, None

    async def _try_stream_candidate(
        self,
        candidate: dict[str, Any],
        request: ChatCompletionRequest,
    ) -> tuple[httpx.Response | None, str | None]:
        name = candidate["name"]
        url, headers, payload = self._prepare_headers_and_payload(
            candidate,
            request,
            stream=True,
        )

        try:
            req = self.client.build_request(
                "POST",
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response = await self.client.send(req, stream=True)
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            self._apply_candidate_failure(candidate)
            err_text = f"Connection error: {str(exc)}"
            logger.warning("Streaming target '%s' request exception: %s", name, err_text)
            return None, err_text

        if response.status_code >= 400:
            body = await response.aread()
            await response.aclose()
            self._apply_candidate_failure(candidate)
            err_msg = f"HTTP {response.status_code}: {body.decode('utf-8', errors='replace')[:300]}"
            logger.warning("Streaming target '%s' failed: %s", name, err_msg)
            return None, err_msg

        return response, None

    async def route_chat_completion(
        self,
        request: ChatCompletionRequest | dict[str, Any],
    ) -> dict[str, Any]:
        """Route a non-streaming Chat Completion request with automatic failover."""
        if isinstance(request, dict):
            request = ChatCompletionRequest.model_validate(request)

        candidates = self.build_candidates_plan(request)
        if not candidates:
            raise NoActiveProvidersError(
                "No available providers: Ollama is disabled and no active cloud providers are configured."
            )

        errors: dict[str, str] = {}
        for candidate in candidates:
            data, err = await self._try_candidate(candidate, request)
            if data is not None:
                return data
            errors[candidate["name"]] = err or "Unknown error"

        raise AllProvidersFailedError(errors)

    async def route_chat_completion_stream(
        self,
        request: ChatCompletionRequest | dict[str, Any],
    ) -> AsyncIterator[bytes]:
        """Route a streaming Chat Completion request with pre-stream failover."""
        if isinstance(request, dict):
            request = ChatCompletionRequest.model_validate(request)

        candidates = self.build_candidates_plan(request)
        if not candidates:
            raise NoActiveProvidersError(
                "No available providers: Ollama is disabled and no active cloud providers are configured."
            )

        errors: dict[str, str] = {}
        for candidate in candidates:
            response, err = await self._try_stream_candidate(candidate, request)
            if response is not None:
                return _stream_response_chunks(response)
            errors[candidate["name"]] = err or "Unknown error"

        raise AllProvidersFailedError(errors)


def _extract_response_text(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    msg = first_choice.get("message")
    if not isinstance(msg, dict):
        return ""
    return str(msg.get("content") or "")


def _extract_tokens_usage(
    data: Any,
    request: ChatCompletionRequest,
) -> tuple[int, int]:
    usage = data.get("usage") if isinstance(data, dict) else None
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None

    if prompt_tokens is None:
        prompt_str = request.get_full_prompt_text()
        prompt_tokens = max(1, len(prompt_str) // 4)

    if completion_tokens is None:
        completion_text = _extract_response_text(data)
        completion_tokens = max(1, len(completion_text) // 4)

    return int(prompt_tokens), int(completion_tokens)


async def _stream_response_chunks(response: httpx.Response) -> AsyncIterator[bytes]:
    try:
        async for chunk in response.aiter_bytes():
            yield chunk
    finally:
        await response.aclose()


# Alias for compatibility
RequestRouter = Router
