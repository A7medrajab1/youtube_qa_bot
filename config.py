"""
Central configuration loaded from environment variables.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ===== AgentRouter Settings =====
AGENT_ROUTER_API_KEY = os.getenv("AGENT_ROUTER_API_KEY")
AGENT_ROUTER_BASE_URL = os.getenv("AGENT_ROUTER_BASE_URL", "https://agentrouter.org/v1")

# ===== OpenRouter Settings =====
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# ===== Default Provider =====
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "openrouter")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "z-ai/glm-4.5-air:free")

# ===== LLM Generation Params =====
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 2000

# ===== Embeddings =====
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ===== Chunking =====
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ===== Retrieval =====
TOP_K = 5

# ===== Provider Configurations =====
PROVIDERS = {
    "agentrouter": {
        "name": "AgentRouter",
        "api_key": AGENT_ROUTER_API_KEY,
        "base_url": AGENT_ROUTER_BASE_URL,
        "needs_spoofing": True,
        "models": [
            "glm-4.5",
            "glm-4.6",
            "glm-5.1",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-6",
            "deepseek-v3.1",
            "deepseek-v3.2",
            "deepseek-r1-0528",
        ],
    },
    "openrouter": {
        "name": "OpenRouter",
        "api_key": OPENROUTER_API_KEY,
        "base_url": OPENROUTER_BASE_URL,
        "needs_spoofing": False,
        "models": [
            "z-ai/glm-4.5-air:free",
        ],
    },
}


def validate_config():
    """Validate that at least one provider is configured."""
    has_agentrouter = bool(AGENT_ROUTER_API_KEY)
    has_openrouter = bool(OPENROUTER_API_KEY)
    
    if not has_agentrouter and not has_openrouter:
        raise ValueError(
            "No provider configured. Set AGENT_ROUTER_API_KEY or OPENROUTER_API_KEY in .env"
        )
    
    return {
        "agentrouter": has_agentrouter,
        "openrouter": has_openrouter,
    }


def get_available_providers() -> list[str]:
    """Return list of providers that have API keys configured."""
    available = []
    for provider_id, cfg in PROVIDERS.items():
        if cfg["api_key"]:
            available.append(provider_id)
    return available