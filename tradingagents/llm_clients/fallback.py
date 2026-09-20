from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.fallbacks import RunnableWithFallbacks
from langchain_core.tools import BaseTool
from rich.console import Console

from .factory import create_llm_client
from .model_catalog import get_fallback_candidates

logger = logging.getLogger("tradingagents.llm_clients.fallback")
console = Console()


def _get_model_name(runnable: Any) -> str:
    """Extract a human-friendly model name from a runnable or chat model."""
    curr = runnable
    while hasattr(curr, "bound"):
        curr = curr.bound
    for attr in ("model_name", "model"):
        val = getattr(curr, attr, None)
        if val and isinstance(val, str):
            return val
    if hasattr(curr, "name") and isinstance(curr.name, str):
        return curr.name
    return curr.__class__.__name__


def is_quota_or_service_unavailable_error(exc: BaseException) -> bool:
    """Check if an exception is due to quota, rate limit, or service unavailability.

    Returns True for:
      - 429 (Too Many Requests, Rate Limit, Resource Exhausted, limit: 0)
      - 503 (Service Unavailable, High Demand)
      - 500, 502, 504, 529 (Temporary server errors / overloads)
      - 404 (When a preview model is retired or not accessible on user's API tier)

    Returns False for:
      - 401 / Authentication errors (invalid API key, unauthorized)
      - ValueError, TypeError, schema validation, bad user input
    """
    if exc is None:
        return False

    exc_type = type(exc).__name__.lower()

    # Never treat authentication errors as quota errors
    if any(k in exc_type for k in ("auth", "unauthoriz", "permissiondenied")):
        return False

    msg = str(exc).lower()
    if any(k in msg for k in ("invalid api key", "unauthorized", "bad authentication", "api key not valid")):
        return False

    # 1. HTTP Status code checks
    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None) or getattr(exc, "http_status", None)
    if status_code in (404, 429, 500, 502, 503, 504, 529):
        return True

    # 2. Known Exception Class Names
    known_classes = {
        "resourceexhausted",
        "toomanyrequests",
        "ratelimiterror",
        "serviceunavailable",
        "internalservererror",
        "apiconnectionerror",
        "badgateway",
        "gatewaytimeout",
        "overloadederror",
    }
    if exc_type in known_classes or any(exc_type.endswith(k) for k in known_classes):
        return True

    # 3. String keywords / message pattern matching
    quota_keywords = (
        "resource_exhausted",
        "resourceexhausted",
        "quota",
        "rate limit",
        "rate_limit",
        "ratelimit",
        "too many requests",
        "429",
        "503",
        "service unavailable",
        "high demand",
        "overloaded",
        "capacity",
        "temporarily unavailable",
        "limit: 0",
        "model is not found",
        "not found for url",
        "server error",
    )
    return any(k in msg for k in quota_keywords)


class QuotaFallbackRunnable(RunnableWithFallbacks):
    """Runnable with fallbacks that only triggers on quota or service availability errors.

    Overrides bind_tools and with_structured_output to ensure that all fallback
    models are wrapped with the exact same tools and schemas across all Python versions.
    """

    def __init__(
        self,
        runnable: Any = None,
        fallbacks: Optional[Sequence[Any]] = None,
        **kwargs: Any,
    ):
        if runnable is not None and "runnable" not in kwargs:
            kwargs["runnable"] = runnable
        if fallbacks is not None and "fallbacks" not in kwargs:
            kwargs["fallbacks"] = list(fallbacks)
        super().__init__(**kwargs)

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, Callable, BaseTool]],
        **kwargs: Any,
    ) -> QuotaFallbackRunnable:
        bound_primary = self.runnable.bind_tools(tools, **kwargs)
        bound_fallbacks = [r.bind_tools(tools, **kwargs) for r in self.fallbacks]
        return self.__class__(
            runnable=bound_primary,
            fallbacks=bound_fallbacks,
            exceptions_to_handle=self.exceptions_to_handle,
            exception_key=self.exception_key,
        )

    def with_structured_output(
        self,
        schema: Union[Dict, type],
        **kwargs: Any,
    ) -> QuotaFallbackRunnable:
        bound_primary = self.runnable.with_structured_output(schema, **kwargs)
        bound_fallbacks = [r.with_structured_output(schema, **kwargs) for r in self.fallbacks]
        return self.__class__(
            runnable=bound_primary,
            fallbacks=bound_fallbacks,
            exceptions_to_handle=self.exceptions_to_handle,
            exception_key=self.exception_key,
        )

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        runnables = list(self.runnables)
        first_error = None
        for idx, runnable in enumerate(runnables):
            try:
                return runnable.invoke(input, config=config, **kwargs)
            except Exception as exc:
                if not is_quota_or_service_unavailable_error(exc):
                    raise
                if first_error is None:
                    first_error = exc
                curr_name = _get_model_name(runnable)
                if idx + 1 < len(runnables):
                    next_name = _get_model_name(runnables[idx + 1])
                    err_msg = str(exc)
                    short_err = (err_msg[:120] + "...") if len(err_msg) > 120 else err_msg
                    logger.warning(
                        "Quota/service error on model '%s': %s. Falling back to '%s'...",
                        curr_name,
                        exc,
                        next_name,
                    )
                    try:
                        console.print(
                            f"[yellow][AVISO] Modelo '{curr_name}' agoto cuota o no esta disponible ({short_err}). "
                            f"Conmutando al modelo de respaldo '{next_name}'...[/yellow]"
                        )
                    except Exception:
                        pass
                    continue
                else:
                    logger.error("All fallback models failed for '%s'. Last error: %s", curr_name, exc)
                    raise
        if first_error is not None:
            raise first_error

    async def ainvoke(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        runnables = list(self.runnables)
        first_error = None
        for idx, runnable in enumerate(runnables):
            try:
                return await runnable.ainvoke(input, config=config, **kwargs)
            except Exception as exc:
                if not is_quota_or_service_unavailable_error(exc):
                    raise
                if first_error is None:
                    first_error = exc
                curr_name = _get_model_name(runnable)
                if idx + 1 < len(runnables):
                    next_name = _get_model_name(runnables[idx + 1])
                    err_msg = str(exc)
                    short_err = (err_msg[:120] + "...") if len(err_msg) > 120 else err_msg
                    logger.warning(
                        "Quota/service error on model '%s': %s. Falling back to '%s'...",
                        curr_name,
                        exc,
                        next_name,
                    )
                    try:
                        console.print(
                            f"[yellow][AVISO] Modelo '{curr_name}' agoto cuota o no esta disponible ({short_err}). "
                            f"Conmutando al modelo de respaldo '{next_name}'...[/yellow]"
                        )
                    except Exception:
                        pass
                    continue
                else:
                    logger.error("All fallback models failed for '%s'. Last error: %s", curr_name, exc)
                    raise
        if first_error is not None:
            raise first_error

    def stream(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Iterator[Any]:
        runnables = list(self.runnables)
        first_error = None
        for idx, runnable in enumerate(runnables):
            try:
                yield from runnable.stream(input, config=config, **kwargs)
                return
            except Exception as exc:
                if not is_quota_or_service_unavailable_error(exc):
                    raise
                if first_error is None:
                    first_error = exc
                curr_name = _get_model_name(runnable)
                if idx + 1 < len(runnables):
                    next_name = _get_model_name(runnables[idx + 1])
                    err_msg = str(exc)
                    short_err = (err_msg[:120] + "...") if len(err_msg) > 120 else err_msg
                    logger.warning(
                        "Quota/service error on model '%s': %s. Falling back to '%s'...",
                        curr_name,
                        exc,
                        next_name,
                    )
                    try:
                        console.print(
                            f"[yellow][AVISO] Modelo '{curr_name}' agoto cuota o no esta disponible ({short_err}). "
                            f"Conmutando al modelo de respaldo '{next_name}'...[/yellow]"
                        )
                    except Exception:
                        pass
                    continue
                else:
                    logger.error("All fallback models failed for '%s'. Last error: %s", curr_name, exc)
                    raise
        if first_error is not None:
            raise first_error


def build_fallback_llm(
    provider: str,
    primary_model: str,
    mode: str = "deep",
    config: Optional[Dict[str, Any]] = None,
    base_url: Optional[str] = None,
    **llm_kwargs: Any,
) -> Any:
    """Build an LLM runnable with automatic fallbacks for quota/service errors.

    Args:
        provider: LLM provider name (e.g. 'google', 'openai', 'anthropic')
        primary_model: The primary model identifier
        mode: 'deep' or 'quick' thinking mode
        config: Optional configuration dictionary
        base_url: Optional API endpoint base URL
        **llm_kwargs: Additional kwargs passed to create_llm_client

    Returns:
        The primary LLM, or a QuotaFallbackRunnable wrapping primary and fallback LLMs.
    """
    primary_client = create_llm_client(
        provider=provider,
        model=primary_model,
        base_url=base_url,
        **llm_kwargs,
    )
    primary_llm = primary_client.get_llm()

    cfg = config or {}
    fallback_enabled = cfg.get("fallback_models_enabled", True)
    if not fallback_enabled:
        return primary_llm

    config_key = "deep_think_fallback_models" if mode == "deep" else "quick_think_fallback_models"
    configured_fallbacks = cfg.get(config_key)

    candidate_names: List[str] = []
    if configured_fallbacks:
        if isinstance(configured_fallbacks, str):
            candidate_names = [m.strip() for m in configured_fallbacks.split(",") if m.strip()]
        elif isinstance(configured_fallbacks, (list, tuple)):
            candidate_names = [str(m).strip() for m in configured_fallbacks if str(m).strip()]
    else:
        candidate_names = get_fallback_candidates(provider, primary_model, mode)

    fallback_names = [m for m in candidate_names if m and m != primary_model]
    if not fallback_names:
        return primary_llm

    fallback_llms = []
    for model_name in fallback_names:
        try:
            fb_client = create_llm_client(
                provider=provider,
                model=model_name,
                base_url=base_url,
                **llm_kwargs,
            )
            fallback_llms.append(fb_client.get_llm())
        except Exception as exc:
            logger.warning(
                "Could not initialize fallback model '%s' for provider '%s': %s",
                model_name,
                provider,
                exc,
            )

    if not fallback_llms:
        return primary_llm

    return QuotaFallbackRunnable(runnable=primary_llm, fallbacks=fallback_llms)

