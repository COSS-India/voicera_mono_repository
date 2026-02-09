# LLM Configuration Mappings
# Default models and available models for each LLM provider

# Available models for each provider
LLM_AVAILABLE_MODELS = {
    "OpenAI": [
        # GPT-5 series (latest)
        "gpt-5.2",
        "gpt-5.1",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        # GPT-4.1 series
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        # GPT-4o series
        "gpt-4o",
        "gpt-4o-mini",
        # Legacy
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
    ],
    "Gemini": [
        # Gemini 3 series (latest - preview)
        "gemini-3.0-pro",
        "gemini-3.0-flash",
        # Gemini 2.5 series (GA)
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ],
    "Kenpath": [None],  # Kenpath doesn't need a model parameter
}

# Default models for each provider
LLM_DEFAULT_MODELS = {
    "OpenAI": "gpt-4o",
    "Gemini": "gemini-2.5-flash",  # Best balance of speed and capability
    "Kenpath": None,
}


def get_llm_model(provider: str, model: str = None) -> str:
    """
    Get the model name for an LLM provider.

    Args:
        provider: LLM provider name (e.g., "OpenAI", "Gemini", "Kenpath")
        model: Optional model override from config

    Returns:
        Model name to use
    """
    # Normalize provider name
    provider_normalized = provider.lower() if provider else ""
    provider_key = {
        "openai": "OpenAI",
        "gemini": "Gemini",
        "google": "Gemini",
        "kenpath": "Kenpath",
    }.get(provider_normalized, provider)

    if model:
        return model

    return LLM_DEFAULT_MODELS.get(provider_key, "gpt-4o")


def get_available_llm_models(provider: str) -> list:
    """
    Get available models for an LLM provider.

    Args:
        provider: LLM provider name

    Returns:
        List of available model names
    """
    provider_normalized = provider.lower() if provider else ""
    provider_key = {
        "openai": "OpenAI",
        "gemini": "Gemini",
        "google": "Gemini",
        "kenpath": "Kenpath",
    }.get(provider_normalized, provider)

    return LLM_AVAILABLE_MODELS.get(provider_key, [])
