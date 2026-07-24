"""
LLM factory: builds the ChatGoogleGenerativeAI (Gemini) client used throughout
the pipeline. Centralized so model swaps only touch this file.

Default: Gemini 1.5 Flash via langchain-google-genai (free tier).
Fallback: OpenAI GPT-4o-mini if OPENAI_API_KEY is set and GOOGLE_API_KEY is not.
"""

from functools import lru_cache

from config import APIKeys, LLMConfig


class LLMNotConfiguredError(RuntimeError):
    """Raised when no LLM API key is available."""


from loguru import logger
from langchain_google_genai import ChatGoogleGenerativeAI

class FallbackChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    fallback_api_key: str = ""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        try:
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except Exception as exc:
            exc_str = str(exc).lower()
            if self.fallback_api_key and self.google_api_key != self.fallback_api_key:
                if any(k in exc_str for k in ["api_key", "api key", "invalid", "unauthorized", "quota", "blocked", "403", "401", "429"]):
                    logger.warning(f"Primary Gemini API key failed. Retrying with fallback key. Error: {exc}")
                    self.google_api_key = self.fallback_api_key
                    self.validate_environment()
                    return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            raise exc

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        try:
            return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except Exception as exc:
            exc_str = str(exc).lower()
            if self.fallback_api_key and self.google_api_key != self.fallback_api_key:
                if any(k in exc_str for k in ["api_key", "api key", "invalid", "unauthorized", "quota", "blocked", "403", "401", "429"]):
                    logger.warning(f"Primary Gemini API key failed. Retrying with fallback key. Error: {exc}")
                    self.google_api_key = self.fallback_api_key
                    self.validate_environment()
                    return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            raise exc


def _build_gemini(temperature: float, model: str = None):
    """Build a Gemini Chat model with fallback API key support."""
    return FallbackChatGoogleGenerativeAI(
        model=model or LLMConfig.MODEL,
        google_api_key=APIKeys.GOOGLE_API_KEY,
        fallback_api_key=APIKeys.GOOGLE_API_KEY_FALLBACK,
        temperature=temperature,
        max_retries=3,
        timeout=60,
    )



def _build_openai(temperature: float, model: str = None):
    """Build an OpenAI Chat model (fallback provider)."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model or "gpt-4o-mini",
        api_key=APIKeys.OPENAI_API_KEY,
        temperature=temperature,
        timeout=60,
    )


def _build_nvidia(temperature: float, model: str = None):
    """Build an NVIDIA Chat model using ChatOpenAI wrapper (OpenAI-compatible)."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model or LLMConfig.MODEL or "meta/llama-3.1-70b-instruct",
        api_key=APIKeys.NVIDIA_API_KEY,
        base_url="https://integrate.api.nvidia.com/v1",
        temperature=temperature,
        timeout=60,
    )


def _build_openrouter(temperature: float, model: str = None):
    """Build an OpenRouter Chat model using ChatOpenAI wrapper (OpenAI-compatible)."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model or LLMConfig.MODEL or "meta-llama/llama-3-70b-instruct",
        api_key=APIKeys.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
        timeout=60,
    )


def _build(provider: str, temperature: float, model: str = None):
    provider = (provider or LLMConfig.PROVIDER).lower()
    if provider == "gemini":
        if not APIKeys.GOOGLE_API_KEY:
            raise LLMNotConfiguredError(
                "GOOGLE_API_KEY is not set. Add it to .env "
                "(see .env.example) or switch LLM_PROVIDER."
            )
        return _build_gemini(temperature, model)
    if provider == "openai":
        if not APIKeys.OPENAI_API_KEY:
            raise LLMNotConfiguredError(
                "OPENAI_API_KEY is not set. Add it to .env."
            )
        return _build_openai(temperature, model)
    if provider == "nvidia":
        if not APIKeys.NVIDIA_API_KEY:
            raise LLMNotConfiguredError(
                "NVIDIA_API_KEY is not set. Add it to .env."
            )
        return _build_nvidia(temperature, model)
    if provider == "openrouter":
        if not APIKeys.OPENROUTER_API_KEY:
            raise LLMNotConfiguredError(
                "OPENROUTER_API_KEY is not set. Add it to .env."
            )
        return _build_openrouter(temperature, model)
    raise ValueError(f"Unknown LLM provider: {provider}")


@lru_cache(maxsize=8)
def get_llm(temperature: float = LLMConfig.TEMPERATURE, model: str = None):
    """Return a cached chat LLM instance.

    Subsequent calls with the same args reuse the instance (reduces setup
    overhead). Different temperatures yield different cached instances.
    """
    return _build(LLMConfig.PROVIDER, temperature, model)


def get_grader_llm():
    """Deterministic LLM for grading / classification nodes."""
    return get_llm(temperature=LLMConfig.GRADER_TEMPERATURE)


def get_generation_llm():
    """LLM for answer generation."""
    return get_llm(temperature=LLMConfig.GENERATION_TEMPERATURE)


def get_hyde_llm():
    """LLM for HyDE hypothetical-document generation (more creative)."""
    return get_llm(temperature=LLMConfig.HYDE_TEMPERATURE)


# ============================================================
# Request-Local Cache & Execution Audit (per single query run)
# ============================================================
import concurrent.futures
import contextvars
import hashlib
import json
import time
from typing import Any, Tuple

_request_llm_cache: contextvars.ContextVar[dict | None] = contextvars.ContextVar("_request_llm_cache", default=None)
_request_audit_log: contextvars.ContextVar[dict | None] = contextvars.ContextVar("_request_audit_log", default=None)


def start_request_context() -> None:
    """Initialize request-local cache and audit trackers for a single query execution."""
    _request_llm_cache.set({})
    _request_audit_log.set({
        "llm_calls": 0,
        "skipped_nodes": [],
        "fallbacks": [],
        "telemetry": [],
    })


def get_request_audit() -> dict:
    audit = _request_audit_log.get()
    if audit is None:
        return {"llm_calls": 0, "skipped_nodes": [], "fallbacks": [], "telemetry": []}
    return audit


def record_audit_event(event_type: str, detail: str, telemetry_data: dict | None = None) -> None:
    audit = _request_audit_log.get()
    if audit is not None:
        if event_type == "llm_call":
            audit["llm_calls"] = audit.get("llm_calls", 0) + 1
        elif event_type == "skip":
            if detail not in audit["skipped_nodes"]:
                audit["skipped_nodes"].append(detail)
        elif event_type == "fallback":
            if detail not in audit["fallbacks"]:
                audit["fallbacks"].append(detail)
        elif event_type == "telemetry" and telemetry_data:
            audit["telemetry"].append(telemetry_data)


# Global shared ThreadPoolExecutor bounded to 16 workers (prevents per-call thread churn)
_GLOBAL_LLM_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=16,
    thread_name_prefix="global_llm_worker"
)


def invoke_chain_safe(chain, inputs: dict, timeout_seconds: float, node_name: str) -> Tuple[Any, bool]:
    """Execute an LLM chain with native request timeout and global shared ThreadPoolExecutor bounds.

    Enforces real timeout bounds so stuck network requests raise TimeoutError immediately.
    Returns (result, was_cached).
    """
    start_t = time.time()
    provider_name = LLMConfig.PROVIDER
    cache = _request_llm_cache.get()
    cache_key = None

    if cache is not None:
        try:
            inputs_str = json.dumps(inputs, sort_keys=True, default=str)
            key_raw = f"{node_name}:{inputs_str}"
            cache_key = hashlib.md5(key_raw.encode("utf-8")).hexdigest()
            if cache_key in cache:
                dur_ms = int((time.time() - start_t) * 1000)
                logger.info(f"[{node_name}] CACHE HIT ({provider_name}, {dur_ms} ms)")
                record_audit_event("telemetry", None, {
                    "node": node_name,
                    "duration_ms": dur_ms,
                    "provider": provider_name,
                    "cached": True,
                    "timeout": False,
                    "fallback": False,
                })
                return cache[cache_key], True
        except Exception:
            cache_key = None

    record_audit_event("llm_call", node_name)

    bound_chain = chain.with_config(config={"timeout": timeout_seconds})

    def _exec():
        return bound_chain.invoke(inputs)

    future = _GLOBAL_LLM_EXECUTOR.submit(_exec)
    try:
        res = future.result(timeout=timeout_seconds)
        dur_ms = int((time.time() - start_t) * 1000)
        if cache is not None and cache_key is not None:
            cache[cache_key] = res
        record_audit_event("telemetry", None, {
            "node": node_name,
            "duration_ms": dur_ms,
            "provider": provider_name,
            "cached": False,
            "timeout": False,
            "fallback": False,
        })
        return res, False
    except concurrent.futures.TimeoutError as exc:
        dur_ms = int((time.time() - start_t) * 1000)
        logger.warning(f"[{node_name}] HARD TIMEOUT after {timeout_seconds}s ({provider_name}, {dur_ms} ms)")
        record_audit_event("telemetry", None, {
            "node": node_name,
            "duration_ms": dur_ms,
            "provider": provider_name,
            "cached": False,
            "timeout": True,
            "fallback": True,
        })
        raise TimeoutError(f"Hard timeout of {timeout_seconds}s exceeded for {node_name}") from exc
    except Exception as exc:
        dur_ms = int((time.time() - start_t) * 1000)
        record_audit_event("telemetry", None, {
            "node": node_name,
            "duration_ms": dur_ms,
            "provider": provider_name,
            "cached": False,
            "timeout": False,
            "fallback": True,
        })
        raise
