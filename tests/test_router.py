import asyncio
from pathlib import Path
import httpx
import pytest
from fastapi.testclient import TestClient

from zero_bill.config import AppConfig, OllamaConfig, ProviderConfig, RoutingConfig
from zero_bill.models import ChatCompletionRequest, ChatMessage
from zero_bill.router import Router
from zero_bill.server import create_app
from zero_bill.tracker import UsageTracker


def test_should_route_to_local() -> None:
    """Test 1: should_route_to_local routes short task to Ollama and large prompt to cloud."""
    config = AppConfig(
        ollama=OllamaConfig(
            enabled=True,
            base_url="http://127.0.0.1:11434/v1",
            model="qwen2.5-coder:7b",
        ),
        routing=RoutingConfig(
            local_max_chars=1200,
            local_keywords=[
                "commit message",
                "summarize",
                "rename",
                "format",
                "typo",
                "docstring",
            ],
        ),
        providers=[],
    )
    router = Router(config)

    short_request = ChatCompletionRequest(
        model="zero-bill-auto",
        messages=[
            ChatMessage(
                role="user",
                content="Please write a commit message for the git staged files",
            )
        ],
    )
    assert router.should_route_to_local(short_request)

    long_content = (
        "Analyze this large software architecture specification and provide a comprehensive "
        "system redesign document covering high availability, partition tolerance, consensus protocols, "
        "cache invalidation strategies, database sharding, and multi-region failover. "
    ) * 10
    assert len(long_content) > 1200

    long_request = ChatCompletionRequest(
        model="zero-bill-auto",
        messages=[ChatMessage(role="user", content=long_content)],
    )
    assert not router.should_route_to_local(long_request)


def test_router_failover_on_429() -> None:
    """Test 2: Provider 1 returns HTTP 429, enters cooldown, and Provider 2 succeeds."""
    config = AppConfig(
        ollama=OllamaConfig(enabled=False),
        providers=[
            ProviderConfig(
                name="provider-primary",
                base_url="https://api.primary.com/v1",
                api_key_env="PRIMARY_KEY",
                model="primary-model",
                priority=1,
                cooldown_seconds=60,
                api_key="key-primary",
                active=True,
                is_active=True,
            ),
            ProviderConfig(
                name="provider-backup",
                base_url="https://api.backup.com/v1",
                api_key_env="BACKUP_KEY",
                model="backup-model",
                priority=2,
                cooldown_seconds=60,
                api_key="key-backup",
                active=True,
                is_active=True,
            ),
        ],
    )

    def mock_handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "api.primary.com" in url_str:
            return httpx.Response(
                429,
                json={"error": {"message": "Rate limit exceeded"}},
            )
        if "api.backup.com" in url_str:
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-test-backup",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": "backup-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Response from provider-backup",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 8,
                        "total_tokens": 20,
                    },
                },
            )
        return httpx.Response(404, json={"error": "Not found"})

    async def _execute_test() -> None:
        mock_client = httpx.AsyncClient(
            transport=httpx.MockTransport(mock_handler)
        )
        router = Router(config=config, client=mock_client)

        request = ChatCompletionRequest(
            model="zero-bill-auto",
            messages=[ChatMessage(role="user", content="Ping test prompt")],
        )

        response = await router.route_chat_completion(request)

        assert router.is_in_cooldown("provider-primary")
        assert not router.is_in_cooldown("provider-backup")

        assert response["choices"][0]["message"]["content"] == "Response from provider-backup"
        assert response.get("_provider") == "provider-backup"

    asyncio.run(_execute_test())


def test_usage_tracker_savings(tmp_path: Path) -> None:
    """Test 3: UsageTracker logs entries and accurately computes estimated_saved_usd."""
    log_file = tmp_path / "usage.jsonl"
    tracker = UsageTracker(log_path=log_file)

    empty_summary = tracker.summary()
    assert empty_summary["total_requests"] == 0
    assert empty_summary["actual_spent_usd"] == 0.0
    assert empty_summary["estimated_saved_usd"] == 0.0

    tracker.record(
        provider="ollama",
        model="qwen2.5-coder:7b",
        route_reason="local",
        prompt_tokens=1000,
        completion_tokens=200,
        actual_cost_usd=0.0,
    )

    tracker.record(
        provider="groq",
        model="llama-3.3-70b-versatile",
        route_reason="cloud",
        prompt_tokens=2000,
        completion_tokens=500,
        actual_cost_usd=0.0,
    )

    summary = tracker.summary()

    assert summary["total_requests"] == 2
    assert summary["local_requests"] == 1
    assert summary["free_api_requests"] == 1
    assert summary["total_tokens"] == 1200 + 2500
    assert summary["actual_spent_usd"] == 0.0

    expected_savings = 0.0195
    assert summary["estimated_saved_usd"] == pytest.approx(expected_savings, rel=1e-4)

    assert "ollama" in summary["by_provider"]
    assert "groq" in summary["by_provider"]
    assert summary["by_provider"]["ollama"]["requests"] == 1
    assert summary["by_provider"]["groq"]["requests"] == 1


def test_e2e_chat_completion_and_tracker(tmp_path: Path) -> None:
    """E2E Test 1: Full POST /v1/chat/completions (stream=False) with failover and usage tracking."""
    config = AppConfig(
        ollama=OllamaConfig(enabled=False),
        providers=[
            ProviderConfig(
                name="prov-fail-429",
                base_url="https://api.fail429.com/v1",
                api_key_env="K1",
                model="model-fail",
                priority=1,
                cooldown_seconds=60,
                api_key="key1",
                active=True,
                is_active=True,
            ),
            ProviderConfig(
                name="prov-success-200",
                base_url="https://api.success200.com/v1",
                api_key_env="K2",
                model="model-success",
                priority=2,
                cooldown_seconds=60,
                api_key="key2",
                active=True,
                is_active=True,
            ),
        ],
    )

    def mock_handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "api.fail429.com" in url_str:
            return httpx.Response(
                429,
                json={"error": {"message": "Rate limit exceeded"}},
            )
        if "api.success200.com" in url_str:
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-e2e-ok",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": "model-success",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "E2E success response",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 10,
                        "total_tokens": 30,
                    },
                },
            )
        return httpx.Response(404, json={"error": "Not found"})

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    tracker = UsageTracker(log_path=tmp_path / "usage.jsonl")
    app = create_app(config, tracker=tracker, client=mock_client)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "zero-bill-auto",
                "messages": [{"role": "user", "content": "Hello e2e test"}],
                "stream": False,
            },
        )
        assert response.status_code == 200
        assert response.headers.get("x-zerobill-provider") == "prov-success-200"

        payload = response.json()
        assert payload["choices"][0]["message"]["content"] == "E2E success response"

        summary = tracker.summary()
        assert summary["total_requests"] == 1
        assert "prov-success-200" in summary["by_provider"]


def test_e2e_streaming_sse() -> None:
    """E2E Test 2: Full POST /v1/chat/completions (stream=True) returns SSE chunks."""
    config = AppConfig(
        ollama=OllamaConfig(enabled=False),
        providers=[
            ProviderConfig(
                name="prov-stream",
                base_url="https://api.streamtest.com/v1",
                api_key_env="K_STREAM",
                model="model-stream",
                priority=1,
                cooldown_seconds=60,
                api_key="key-stream",
                active=True,
                is_active=True,
            ),
        ],
    )

    def mock_stream_handler(request: httpx.Request) -> httpx.Response:
        sse_data = (
            b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
            b"data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=sse_data,
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_stream_handler))
    app = create_app(config, client=mock_client)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "zero-bill-auto",
                "messages": [{"role": "user", "content": "Hello stream"}],
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        assert "hello" in response.text
        assert "[DONE]" in response.text
