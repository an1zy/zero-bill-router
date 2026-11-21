from typing import Any
from pydantic import BaseModel, ConfigDict, Field


def _extract_part_text(part: Any) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, dict) and "text" in part:
        return str(part["text"])
    return ""


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[Any] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

    def get_text_content(self) -> str:
        if isinstance(self.content, str):
            return self.content
        if not isinstance(self.content, list):
            return ""
        parts = [_extract_part_text(item) for item in self.content]
        return " ".join(p for p in parts if p)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool | None = False
    top_p: float | None = None
    n: int | None = 1
    stop: str | list[str] | None = None
    max_completion_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None

    def get_total_chars(self) -> int:
        return sum(len(msg.get_text_content()) for msg in self.messages)

    def get_full_prompt_text(self) -> str:
        return " ".join(msg.get_text_content() for msg in self.messages)


class ChatCompletionChoiceMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str = "assistant"
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionChoice(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int = 0
    message: ChatCompletionChoiceMessage
    finish_reason: str | None = "stop"


class UsageInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo | None = None
