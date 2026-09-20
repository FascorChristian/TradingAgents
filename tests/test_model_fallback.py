"""Tests for automatic LLM fallback on quota and service unavailable errors."""

from __future__ import annotations

from typing import Any, List, Optional
import pytest
from pydantic import BaseModel

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from tradingagents.llm_clients.fallback import (
    QuotaFallbackRunnable,
    _get_model_name,
    build_fallback_llm,
    is_quota_or_service_unavailable_error,
)
from tradingagents.llm_clients.model_catalog import get_fallback_candidates


# ---------------------------------------------------------------------------
# Test Dummies and Exceptions
# ---------------------------------------------------------------------------

class FakeHTTPStatusError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class FakeResourceExhausted(Exception):
    pass


class FakeServiceUnavailable(Exception):
    pass


class FakeRateLimitError(Exception):
    pass


class FakeAuthError(Exception):
    def __init__(self, message: str = "Invalid API key"):
        super().__init__(message)
        self.status_code = 401


class DummyStructuredOutput(BaseModel):
    summary: str


class MockChatModel(BaseChatModel):
    """Configurable mock chat model for fallback testing."""

    model_id: str
    failure_exc: Optional[Exception] = None
    response_text: str = "mock output"

    @property
    def _llm_type(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self.model_id

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.failure_exc is not None:
            raise self.failure_exc
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.response_text))]
        )

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        bound = MockChatModel(
            model_id=f"{self.model_id}-bound-tools",
            failure_exc=self.failure_exc,
            response_text=f"{self.response_text}-with-tools",
        )
        return bound

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        bound = MockChatModel(
            model_id=f"{self.model_id}-structured",
            failure_exc=self.failure_exc,
            response_text="structured-ok",
        )
        return bound


# ---------------------------------------------------------------------------
# Unit Tests: is_quota_or_service_unavailable_error
# ---------------------------------------------------------------------------

def test_is_quota_error_status_codes():
    assert is_quota_or_service_unavailable_error(FakeHTTPStatusError("Rate limited", 429))
    assert is_quota_or_service_unavailable_error(FakeHTTPStatusError("Unavailable", 503))
    assert is_quota_or_service_unavailable_error(FakeHTTPStatusError("Bad gateway", 502))
    assert is_quota_or_service_unavailable_error(FakeHTTPStatusError("Server error", 500))
    assert is_quota_or_service_unavailable_error(FakeHTTPStatusError("Gateway timeout", 504))
    assert is_quota_or_service_unavailable_error(FakeHTTPStatusError("Overloaded", 529))
    assert is_quota_or_service_unavailable_error(FakeHTTPStatusError("Model retired / not found", 404))


def test_is_quota_error_exception_classes():
    assert is_quota_or_service_unavailable_error(FakeResourceExhausted("quota"))
    assert is_quota_or_service_unavailable_error(FakeServiceUnavailable("service down"))
    assert is_quota_or_service_unavailable_error(FakeRateLimitError("too fast"))


def test_is_quota_error_message_patterns():
    assert is_quota_or_service_unavailable_error(Exception("Resource has been exhausted (e.g. check quota). limit: 0"))
    assert is_quota_or_service_unavailable_error(Exception("The service is currently unavailable due to high demand"))
    assert is_quota_or_service_unavailable_error(Exception("Rate limit reached for requests per minute"))
    assert is_quota_or_service_unavailable_error(Exception("Too Many Requests"))
    assert is_quota_or_service_unavailable_error(Exception("The model is overloaded. Please try again later."))


def test_is_quota_error_excludes_auth_and_bad_input():
    assert not is_quota_or_service_unavailable_error(FakeAuthError())
    assert not is_quota_or_service_unavailable_error(Exception("Invalid API key provided"))
    assert not is_quota_or_service_unavailable_error(Exception("Unauthorized access"))
    assert not is_quota_or_service_unavailable_error(ValueError("Invalid argument: foo"))
    assert not is_quota_or_service_unavailable_error(TypeError("unsupported operand type"))
    assert not is_quota_or_service_unavailable_error(None)


# ---------------------------------------------------------------------------
# Unit Tests: get_fallback_candidates
# ---------------------------------------------------------------------------

def test_get_fallback_candidates_google_deep():
    candidates = get_fallback_candidates("google", "gemini-3.1-pro-preview", "deep")
    assert candidates == ["gemini-3.5-flash", "gemini-3.1-flash-lite"]


def test_get_fallback_candidates_google_quick():
    candidates = get_fallback_candidates("google", "gemini-3.5-flash", "quick")
    assert candidates == ["gemini-3.1-flash-lite"]


def test_get_fallback_candidates_google_flash_as_deep():
    candidates = get_fallback_candidates("google", "gemini-3.5-flash", "deep")
    assert candidates == ["gemini-3.1-flash-lite", "gemini-3.1-pro-preview"]


def test_get_fallback_candidates_openai():
    candidates = get_fallback_candidates("openai", "gpt-5.6", "deep")
    assert "gpt-5.6-terra" in candidates
    assert "gpt-5.6" not in candidates


def test_get_fallback_candidates_unknown_provider():
    assert get_fallback_candidates("nonexistent", "model", "deep") == []


# ---------------------------------------------------------------------------
# Unit Tests: QuotaFallbackRunnable
# ---------------------------------------------------------------------------

def test_quota_fallback_primary_succeeds():
    primary = MockChatModel(model_id="primary", response_text="primary response")
    fallback = MockChatModel(model_id="fallback", response_text="fallback response")

    qfr = QuotaFallbackRunnable(runnable=primary, fallbacks=[fallback])
    res = qfr.invoke("hello")
    assert res.content == "primary response"


def test_quota_fallback_triggers_on_429():
    primary = MockChatModel(
        model_id="primary",
        failure_exc=FakeHTTPStatusError("Resource exhausted: limit: 0", 429),
    )
    fallback = MockChatModel(model_id="fallback", response_text="fallback response")

    qfr = QuotaFallbackRunnable(runnable=primary, fallbacks=[fallback])
    res = qfr.invoke("hello")
    assert res.content == "fallback response"


def test_quota_fallback_multiple_chain():
    m1 = MockChatModel(
        model_id="m1",
        failure_exc=FakeHTTPStatusError("Resource exhausted", 429),
    )
    m2 = MockChatModel(
        model_id="m2",
        failure_exc=FakeHTTPStatusError("Service unavailable: high demand", 503),
    )
    m3 = MockChatModel(model_id="m3", response_text="m3 success")

    qfr = QuotaFallbackRunnable(runnable=m1, fallbacks=[m2, m3])
    res = qfr.invoke("hello")
    assert res.content == "m3 success"


def test_quota_fallback_does_not_catch_value_error():
    primary = MockChatModel(
        model_id="primary",
        failure_exc=ValueError("Invalid parameter"),
    )
    fallback = MockChatModel(model_id="fallback", response_text="fallback response")

    qfr = QuotaFallbackRunnable(runnable=primary, fallbacks=[fallback])
    with pytest.raises(ValueError, match="Invalid parameter"):
        qfr.invoke("hello")


def test_quota_fallback_bind_tools_propagates():
    @tool
    def sample_tool(query: str) -> str:
        """Sample tool for testing."""
        return query

    primary = MockChatModel(
        model_id="primary",
        failure_exc=FakeHTTPStatusError("Quota exceeded", 429),
    )
    fallback = MockChatModel(model_id="fallback", response_text="fallback response")

    qfr = QuotaFallbackRunnable(runnable=primary, fallbacks=[fallback])
    bound = qfr.bind_tools([sample_tool])

    assert isinstance(bound, QuotaFallbackRunnable)
    res = bound.invoke("hello")
    assert res.content == "fallback response-with-tools"


def test_quota_fallback_with_structured_output_propagates():
    primary = MockChatModel(
        model_id="primary",
        failure_exc=FakeHTTPStatusError("Quota exceeded", 429),
    )
    fallback = MockChatModel(model_id="fallback", response_text="fallback response")

    qfr = QuotaFallbackRunnable(runnable=primary, fallbacks=[fallback])
    structured = qfr.with_structured_output(DummyStructuredOutput)

    assert isinstance(structured, QuotaFallbackRunnable)
    res = structured.invoke("hello")
    assert res.content == "structured-ok"


def test_quota_fallback_all_fail_raises_last_error():
    m1 = MockChatModel(
        model_id="m1",
        failure_exc=FakeHTTPStatusError("Resource exhausted", 429),
    )
    m2 = MockChatModel(
        model_id="m2",
        failure_exc=FakeHTTPStatusError("Service unavailable", 503),
    )

    qfr = QuotaFallbackRunnable(runnable=m1, fallbacks=[m2])
    with pytest.raises(FakeHTTPStatusError) as exc_info:
        qfr.invoke("hello")
    assert exc_info.value.status_code == 503


# ---------------------------------------------------------------------------
# Unit Tests: build_fallback_llm
# ---------------------------------------------------------------------------

def test_build_fallback_llm_disabled(monkeypatch):
    from tradingagents.llm_clients.base_client import BaseLLMClient

    class MockClient(BaseLLMClient):
        def get_llm(self):
            return MockChatModel(model_id=self.model)

        def validate_model(self):
            return True

    monkeypatch.setattr("tradingagents.llm_clients.fallback.create_llm_client", lambda **k: MockClient(k["model"]))

    config = {"fallback_models_enabled": False}
    llm = build_fallback_llm(
        provider="google",
        primary_model="gemini-3.1-pro-preview",
        mode="deep",
        config=config,
    )
    # When disabled, should return raw MockChatModel rather than QuotaFallbackRunnable
    assert isinstance(llm, MockChatModel)
    assert llm.model_id == "gemini-3.1-pro-preview"


def test_build_fallback_llm_enabled_auto(monkeypatch):
    from tradingagents.llm_clients.base_client import BaseLLMClient

    class MockClient(BaseLLMClient):
        def get_llm(self):
            return MockChatModel(model_id=self.model)

        def validate_model(self):
            return True

    monkeypatch.setattr("tradingagents.llm_clients.fallback.create_llm_client", lambda **k: MockClient(k["model"]))

    config = {"fallback_models_enabled": True}
    llm = build_fallback_llm(
        provider="google",
        primary_model="gemini-3.1-pro-preview",
        mode="deep",
        config=config,
    )
    assert isinstance(llm, QuotaFallbackRunnable)
    assert _get_model_name(llm.runnable) == "gemini-3.1-pro-preview"
    fallback_names = [_get_model_name(r) for r in llm.fallbacks]
    assert fallback_names == ["gemini-3.5-flash", "gemini-3.1-flash-lite"]


def test_build_fallback_llm_custom_list(monkeypatch):
    from tradingagents.llm_clients.base_client import BaseLLMClient

    class MockClient(BaseLLMClient):
        def get_llm(self):
            return MockChatModel(model_id=self.model)

        def validate_model(self):
            return True

    monkeypatch.setattr("tradingagents.llm_clients.fallback.create_llm_client", lambda **k: MockClient(k["model"]))

    config = {
        "fallback_models_enabled": True,
        "deep_think_fallback_models": "custom-fb-1, custom-fb-2",
    }
    llm = build_fallback_llm(
        provider="google",
        primary_model="gemini-3.1-pro-preview",
        mode="deep",
        config=config,
    )
    assert isinstance(llm, QuotaFallbackRunnable)
    fallback_names = [_get_model_name(r) for r in llm.fallbacks]
    assert fallback_names == ["custom-fb-1", "custom-fb-2"]


def test_trading_graph_initializes_with_fallbacks(monkeypatch, tmp_path):
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.llm_clients.base_client import BaseLLMClient

    class MockClient(BaseLLMClient):
        def get_llm(self):
            return MockChatModel(model_id=self.model)

        def validate_model(self):
            return True

    monkeypatch.setattr("tradingagents.llm_clients.fallback.create_llm_client", lambda **k: MockClient(k["model"]))

    config = dict(DEFAULT_CONFIG)
    config["results_dir"] = str(tmp_path / "results")
    config["data_cache_dir"] = str(tmp_path / "cache")
    config["memory_log_path"] = str(tmp_path / "memory.md")
    config["llm_provider"] = "google"
    config["deep_think_llm"] = "gemini-3.1-pro-preview"
    config["quick_think_llm"] = "gemini-3.5-flash"
    config["fallback_models_enabled"] = True

    graph = TradingAgentsGraph(selected_analysts=["market"], config=config)

    assert isinstance(graph.deep_thinking_llm, QuotaFallbackRunnable)
    assert _get_model_name(graph.deep_thinking_llm.runnable) == "gemini-3.1-pro-preview"
    assert [_get_model_name(r) for r in graph.deep_thinking_llm.fallbacks] == ["gemini-3.5-flash", "gemini-3.1-flash-lite"]

    assert isinstance(graph.quick_thinking_llm, QuotaFallbackRunnable)
    assert _get_model_name(graph.quick_thinking_llm.runnable) == "gemini-3.5-flash"
    assert [_get_model_name(r) for r in graph.quick_thinking_llm.fallbacks] == ["gemini-3.1-flash-lite"]
