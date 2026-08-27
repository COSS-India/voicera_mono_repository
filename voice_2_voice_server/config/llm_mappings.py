# LLM Configuration Mappings
# Default models for each LLM provider

LLM_DEFAULT_MODELS = {
    "OpenAI": "gpt-4o",
    "Kenpath": None,  # Kenpath doesn't need a model parameter
    "anthropic": "claude-sonnet-4-5-20250929",
    "Grok": "grok-3-beta",
    "grok": "grok-3-beta",
    # Must match vLLM's --served-model-name, pinned in llm/<model>/Dockerfile.
    "qwen": "qwen3.5-4b",
    "custom_llm": None,
}


def get_llm_model(provider: str, model: str = None) -> str:
    """
    Get the model name for an LLM provider
    
    Args:
        provider: LLM provider name (e.g., "openai", "kenpath")
        model: Optional model override from config
    
    Returns:
        Model name to use
    """
    provider = provider.lower()
    
    if model:
        return model

    default = LLM_DEFAULT_MODELS.get(provider)
    if default is None and provider == "custom_llm":
        return model or ""
    
    return LLM_DEFAULT_MODELS.get(provider, "gpt-4o")
