# LLM Configuration Mappings
# Default models for each LLM provider

LLM_DEFAULT_MODELS = {
    "OpenAI": "gpt-4o",
    "Kenpath": None,  # Kenpath doesn't need a model parameter
    "anthropic": "claude-sonnet-4-5-20250929",
    "Grok": "grok-3-beta",
    "grok": "grok-3-beta",
    "qwen": "Qwen/Qwen3-8B",
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
    
    return LLM_DEFAULT_MODELS.get(provider, "gpt-4o")
