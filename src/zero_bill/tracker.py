import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class UsageTracker:
    """Tracks token usage, actual costs, and estimated dollar savings in a JSONL log."""

    BASELINE_INPUT_COST_PER_M = 3.00
    BASELINE_OUTPUT_COST_PER_M = 15.00

    def __init__(
        self,
        log_path: Path | str = Path.home() / ".zero-bill" / "usage.jsonl",
    ) -> None:
        self.log_path = Path(log_path)

    def record(
        self,
        provider: str,
        model: str,
        route_reason: str,
        prompt_tokens: int,
        completion_tokens: int,
        actual_cost_usd: float = 0.0,
    ) -> dict[str, Any]:
        """Record a completed request with token counts, cost, and savings calculation."""
        prompt_tokens = max(0, int(prompt_tokens))
        completion_tokens = max(0, int(completion_tokens))
        actual_cost = max(0.0, float(actual_cost_usd))

        # Flagship model tariff: $3.00/1M input + $15.00/1M output
        baseline_cost = (
            (prompt_tokens * self.BASELINE_INPUT_COST_PER_M)
            + (completion_tokens * self.BASELINE_OUTPUT_COST_PER_M)
        ) / 1_000_000.0

        saved_usd = baseline_cost - actual_cost

        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model": model,
            "route_reason": route_reason,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "actual_cost_usd": round(actual_cost, 6),
            "baseline_cost_usd": round(baseline_cost, 6),
            "saved_usd": round(saved_usd, 6),
        }

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        return entry

    def summary(self) -> dict[str, Any]:
        """Read the JSONL log and compute aggregated usage and savings metrics."""
        if not self.log_path.exists():
            return {
                "total_requests": 0,
                "local_requests": 0,
                "free_api_requests": 0,
                "total_tokens": 0,
                "actual_spent_usd": 0.0,
                "estimated_saved_usd": 0.0,
                "by_provider": {},
            }

        total_requests = 0
        local_requests = 0
        free_api_requests = 0
        total_tokens = 0
        actual_spent_usd = 0.0
        estimated_saved_usd = 0.0
        by_provider: dict[str, dict[str, Any]] = {}

        with self.log_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    item = json.loads(stripped)
                except json.JSONDecodeError:
                    continue

                provider = item.get("provider", "unknown")
                p_tokens = int(item.get("prompt_tokens", 0))
                c_tokens = int(item.get("completion_tokens", 0))
                item_total_tokens = int(item.get("total_tokens", p_tokens + c_tokens))
                cost = float(item.get("actual_cost_usd", 0.0))
                saved = float(item.get("saved_usd", 0.0))
                reason = item.get("route_reason", "")

                total_requests += 1
                total_tokens += item_total_tokens
                actual_spent_usd += cost
                estimated_saved_usd += saved

                is_local = (
                    provider.lower() in ("ollama", "local")
                    or "local" in reason.lower()
                )
                if is_local:
                    local_requests += 1
                elif cost == 0.0:
                    free_api_requests += 1

                if provider not in by_provider:
                    by_provider[provider] = {
                        "requests": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "actual_spent_usd": 0.0,
                        "saved_usd": 0.0,
                    }

                p_stats = by_provider[provider]
                p_stats["requests"] += 1
                p_stats["prompt_tokens"] += p_tokens
                p_stats["completion_tokens"] += c_tokens
                p_stats["total_tokens"] += item_total_tokens
                p_stats["actual_spent_usd"] = round(p_stats["actual_spent_usd"] + cost, 6)
                p_stats["saved_usd"] = round(p_stats["saved_usd"] + saved, 6)

        return {
            "total_requests": total_requests,
            "local_requests": local_requests,
            "free_api_requests": free_api_requests,
            "total_tokens": total_tokens,
            "actual_spent_usd": round(actual_spent_usd, 6),
            "estimated_saved_usd": round(estimated_saved_usd, 6),
            "by_provider": by_provider,
        }
