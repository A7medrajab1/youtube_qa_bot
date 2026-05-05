"""
LLM initialization with multi-provider support (AgentRouter + OpenRouter).
"""
import uuid
from langchain_openai import ChatOpenAI
from config import PROVIDERS, LLM_TEMPERATURE, LLM_MAX_TOKENS


def _spoofed_headers() -> dict:
    """Codex CLI headers required by AgentRouter."""
    return {
        "User-Agent": "codex_cli_rs/0.21.0 (Linux; x86_64) Cursor/1.0.0",
        "OpenAI-Beta": "responses=experimental",
        "originator": "codex_cli_rs",
        "session_id": str(uuid.uuid4()),
        "version": "0.21.0",
    }


def _openrouter_headers() -> dict:
    """Optional headers OpenRouter recommends for tracking/rankings."""
    return {
        "HTTP-Referer": "http://localhost:7860",
        "X-Title": "YouTube QA Bot",
    }


def get_llm(provider: str, model: str) -> ChatOpenAI:
    """
    Initialize a ChatOpenAI client for the specified provider and model.
    
    Args:
        provider: "agentrouter" or "openrouter"
        model: Model name (must be in the provider's supported list)
    
    Returns:
        Configured ChatOpenAI instance.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(PROVIDERS.keys())}")

    cfg = PROVIDERS[provider]

    if not cfg["api_key"]:
        raise ValueError(
            f"{cfg['name']} API key not configured. Set {provider.upper()}_API_KEY in .env"
        )

    # Choose headers based on provider
    if cfg["needs_spoofing"]:
        headers = _spoofed_headers()
    else:
        headers = _openrouter_headers()

    return ChatOpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        model=model,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        default_headers=headers,
        timeout=120.0,
    )