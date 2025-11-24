"""
Model factory for creating LLM instances based on environment configuration.

This module provides a centralized way to create LLM instances for either
OpenAI or Google Gemini models based on the LLM_PROVIDER environment variable.
"""

import os
import logging
from typing import Literal, Optional
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    "openai": {
        "analysis": {
            "model": "gpt-4o",
            "temperature": 0.1,
            "max_tokens": 400,
        },
        "research": {
            "model": "gpt-4o-mini",
            "temperature": 0.2,
            "max_tokens": 400,
        },
    },
    "gemini": {
        "analysis": {
            "model": "gemini-2.0-flash-exp",
            "temperature": 0.1,
            "max_tokens": 400,
        },
        "research": {
            "model": "gemini-2.0-flash-exp",
            "temperature": 0.2,
            "max_tokens": 400,
        },
    },
}


def get_llm_provider() -> str:
    """
    Get the LLM provider from environment variable.
    
    Returns:
        str: The LLM provider ('openai' or 'gemini'), defaults to 'openai'
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider not in ["openai", "gemini"]:
        logger.warning(
            f"Invalid LLM_PROVIDER '{provider}', defaulting to 'openai'. "
            "Valid options are: 'openai', 'gemini'"
        )
        return "openai"
    return provider


def validate_api_key(provider: str) -> bool:
    """
    Validate that the required API key is set for the provider.
    
    Args:
        provider: The LLM provider ('openai' or 'gemini')
        
    Returns:
        bool: True if API key is set, False otherwise
    """
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error(
                "OPENAI_API_KEY not found in environment. "
                "Please set it in your .env file or environment variables."
            )
            return False
    elif provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error(
                "GOOGLE_API_KEY not found in environment. "
                "Please set it in your .env file or environment variables."
            )
            return False
    return True


def get_llm(
    llm_type: Literal["analysis", "research"] = "analysis",
    provider: Optional[str] = None,
    **kwargs
) -> BaseChatModel:
    """
    Get an LLM instance based on the provider and type.
    
    Args:
        llm_type: Type of LLM to create ('analysis' or 'research')
        provider: LLM provider ('openai' or 'gemini'). If None, uses LLM_PROVIDER env var
        **kwargs: Additional arguments to override default configuration
        
    Returns:
        BaseChatModel: Configured LLM instance
        
    Raises:
        ValueError: If API key is not set for the selected provider
        
    Examples:
        >>> # Use default provider from environment
        >>> llm = get_llm("analysis")
        
        >>> # Explicitly use OpenAI
        >>> llm = get_llm("research", provider="openai")
        
        >>> # Use Gemini with custom temperature
        >>> llm = get_llm("analysis", provider="gemini", temperature=0.5)
    """
    # Determine provider
    if provider is None:
        provider = get_llm_provider()
    
    # Validate API key
    if not validate_api_key(provider):
        raise ValueError(
            f"API key not set for provider '{provider}'. "
            f"Please set {'OPENAI_API_KEY' if provider == 'openai' else 'GOOGLE_API_KEY'} "
            "in your .env file or environment variables."
        )
    
    # Get configuration for this provider and type
    config = MODEL_CONFIGS[provider][llm_type].copy()
    config.update(kwargs)  # Allow overrides
    
    # Create LLM instance
    if provider == "openai":
        logger.info(f"Creating OpenAI {llm_type} LLM with model {config['model']}")
        return ChatOpenAI(**config)
    elif provider == "gemini":
        logger.info(f"Creating Gemini {llm_type} LLM with model {config['model']}")
        # Gemini uses different parameter names
        gemini_config = {
            "model": config["model"],
            "temperature": config["temperature"],
        }
        # Gemini uses max_output_tokens instead of max_tokens
        if "max_tokens" in config:
            gemini_config["max_output_tokens"] = config["max_tokens"]
        return ChatGoogleGenerativeAI(**gemini_config)
    
    raise ValueError(f"Unsupported provider: {provider}")


def get_analysis_llm(provider: Optional[str] = None, **kwargs) -> BaseChatModel:
    """
    Get an analysis LLM instance (higher quality, used for decomposition and synthesis).
    
    Args:
        provider: LLM provider ('openai' or 'gemini'). If None, uses LLM_PROVIDER env var
        **kwargs: Additional arguments to override default configuration
        
    Returns:
        BaseChatModel: Configured analysis LLM instance
    """
    return get_llm("analysis", provider=provider, **kwargs)


def get_research_llm(provider: Optional[str] = None, **kwargs) -> BaseChatModel:
    """
    Get a research LLM instance (optimized for research tasks).
    
    Args:
        provider: LLM provider ('openai' or 'gemini'). If None, uses LLM_PROVIDER env var
        **kwargs: Additional arguments to override default configuration
        
    Returns:
        BaseChatModel: Configured research LLM instance
    """
    return get_llm("research", provider=provider, **kwargs)
