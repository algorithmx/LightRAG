from ..utils import verbose_debug, VERBOSE_DEBUG
import sys
import os
import logging
import numpy as np
from typing import Any, Union, AsyncIterator
import pipmaster as pm  # Pipmaster for dynamic library install

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator

# Install OpenAI SDK if not present (since xAI API is OpenAI-compatible)
if not pm.is_installed("openai"):
    pm.install("openai")

from openai import (
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
    InvalidResponseError,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from lightrag.utils import (
    safe_unicode_decode,
    logger,
    wrap_embedding_func_with_attrs,
    locate_json_string_body_from_string,
)
from lightrag.api import __api_version__


def create_grok_async_client(
    api_key: str | None = None,
    base_url: str | None = None,
    client_configs: dict[str, Any] | None = None,
) -> AsyncOpenAI:
    """Create an async xAI Grok client using OpenAI SDK."""
    if client_configs is None:
        client_configs = {}
    
    if not api_key:
        api_key = os.environ.get("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY environment variable is required")
    
    if not base_url:
        base_url = "https://api.x.ai/v1"
    
    default_headers = {
        "User-Agent": f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_8) LightRAG/{__api_version__}",
        "Content-Type": "application/json",
    }
    
    # Set logger level to INFO when VERBOSE_DEBUG is off
    if not VERBOSE_DEBUG and logger.level == logging.DEBUG:
        logging.getLogger("openai").setLevel(logging.INFO)
    
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers=default_headers,
        **client_configs,
    )


# Core xAI Grok completion function with retry
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=(
        retry_if_exception_type(RateLimitError)
        | retry_if_exception_type(APIConnectionError)
        | retry_if_exception_type(APITimeoutError)
        | retry_if_exception_type(InvalidResponseError)
    ),
)
async def grok_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    token_tracker: Any | None = None,
    **kwargs: Any,
) -> str:
    """Complete a prompt using xAI Grok API with caching support.
    
    Args:
        model: The Grok model to use (e.g., 'grok-4-0709', 'grok-3', 'grok-3-mini')
        prompt: The user prompt
        system_prompt: Optional system prompt
        history_messages: Optional conversation history
        base_url: Optional custom base URL (defaults to https://api.x.ai/v1)
        api_key: Optional API key (defaults to XAI_API_KEY env var)
        token_tracker: Optional token usage tracker
        **kwargs: Additional parameters for the API call
        
    Returns:
        The completion text from Grok
    """
    if history_messages is None:
        history_messages = []
    
    # Create the Grok client
    grok_async_client = create_grok_async_client(
        api_key=api_key, base_url=base_url, client_configs=kwargs.pop("client_configs", {})
    )
    
    kwargs.pop("hashing_kv", None)
    
    # Build messages array
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    logger.debug("===== Sending Query to xAI Grok LLM =====")
    logger.debug(f"Model: {model}   Base URL: {base_url}")
    logger.debug(f"Additional kwargs: {kwargs}")
    verbose_debug(f"Query: {prompt}")
    verbose_debug(f"System prompt: {system_prompt}")
    
    try:
        # Don't use async with context manager, use client directly
        if "response_format" in kwargs:
            response = await grok_async_client.beta.chat.completions.parse(
                model=model, messages=messages, **kwargs
            )
        else:
            response = await grok_async_client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
        
        # Track token usage if tracker is provided
        if token_tracker is not None and hasattr(response, "usage"):
            if hasattr(response.usage, "prompt_tokens"):
                token_tracker.prompt_tokens += response.usage.prompt_tokens
            if hasattr(response.usage, "completion_tokens"):
                token_tracker.completion_tokens += response.usage.completion_tokens
        
        # Extract content from response
        content = response.choices[0].message.content
        if content is None:
            logger.warning("Received None content from Grok API")
            return ""
        
        # Handle unicode decoding if needed
        if r"\u" in content:
            content = safe_unicode_decode(content.encode("utf-8"))
        
        logger.debug(f"Grok API response received, length: {len(content)}")
        verbose_debug(f"Response: {content}")
        
        return content
        
    except APIConnectionError as e:
        logger.error(f"xAI Grok API Connection Error: {e}")
        raise
    except RateLimitError as e:
        logger.error(f"xAI Grok API Rate Limit Error: {e}")
        raise
    except APITimeoutError as e:
        logger.error(f"xAI Grok API Timeout Error: {e}")
        raise
    except Exception as e:
        logger.error(
            f"xAI Grok API Call Failed,\nModel: {model},\nParams: {kwargs}, Got: {e}"
        )
        raise
    finally:
        await grok_async_client.close()


# Generic Grok completion function
async def grok_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    keyword_extraction: bool = False,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    """Generic Grok completion function that uses the configured model."""
    if history_messages is None:
        history_messages = []
    
    keyword_extraction = kwargs.pop("keyword_extraction", keyword_extraction)
    if keyword_extraction:
        kwargs["response_format"] = {"type": "json_object"}
    
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    result = await grok_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )
    
    if keyword_extraction:
        return locate_json_string_body_from_string(result)
    return result


# Grok 4 specific completion
async def grok_4_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    """Grok 4 specific completion function."""
    if history_messages is None:
        history_messages = []
    return await grok_complete_if_cache(
        "grok-4-0709",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


# Grok 3 specific completion
async def grok_3_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    """Grok 3 specific completion function."""
    if history_messages is None:
        history_messages = []
    return await grok_complete_if_cache(
        "grok-3",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


# Grok 3 Mini specific completion
async def grok_3_mini_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    """Grok 3 Mini specific completion function."""
    if history_messages is None:
        history_messages = []
    return await grok_complete_if_cache(
        "grok-3-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


# Grok 3 Fast specific completion
async def grok_3_fast_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    """Grok 3 Fast specific completion function."""
    if history_messages is None:
        history_messages = []
    return await grok_complete_if_cache(
        "grok-3-fast",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


# Grok 3 Mini Fast specific completion
async def grok_3_mini_fast_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    """Grok 3 Mini Fast specific completion function."""
    if history_messages is None:
        history_messages = []
    return await grok_complete_if_cache(
        "grok-3-mini-fast",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


# Grok 2 Vision specific completion
async def grok_2_vision_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    """Grok 2 Vision specific completion function."""
    if history_messages is None:
        history_messages = []
    return await grok_complete_if_cache(
        "grok-2-vision-1212",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


# Embedding function (xAI doesn't provide native embeddings, so we use OpenAI as fallback)
@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8191)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def grok_embed(
    texts: list[str],
    model: str = "text-embedding-3-small",  # Default to OpenAI embedding since xAI doesn't provide embeddings
    base_url: str = None,
    api_key: str = None,
) -> np.ndarray:
    """
    Generate embeddings using OpenAI API since xAI doesn't provide native embedding support.

    Note: This function uses OpenAI's embedding API as a fallback since xAI Grok doesn't
    provide embedding models. You'll need an OpenAI API key in addition to your xAI API key.

    Args:
        texts: List of text strings to embed
        model: OpenAI embedding model name (e.g., "text-embedding-3-small", "text-embedding-3-large")
        base_url: Optional custom base URL (not used, kept for compatibility)
        api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)

    Returns:
        numpy array of shape (len(texts), embedding_dimension) containing the embeddings
    """
    raise NotImplementedError("xAI Grok doesn't provide embedding models.")


# Helper function to get available Grok models
def get_available_grok_models() -> dict[str, dict]:
    """
    Returns a dictionary of available xAI Grok models and their properties.
    """
    return {
        "grok-4-0709": {
            "context_length": 256000,
            "capabilities": ["text", "vision", "function_calling", "structured_outputs", "reasoning"],
            "description": "Latest Grok 4 model with advanced reasoning capabilities",
            "pricing": {"input": 3.00, "output": 15.00},  # per million tokens
        },
        "grok-3": {
            "context_length": 131072,
            "capabilities": ["text", "function_calling", "structured_outputs"],
            "description": "Grok 3 general purpose model",
            "pricing": {"input": 3.00, "output": 15.00},
        },
        "grok-3-mini": {
            "context_length": 131072,
            "capabilities": ["text", "function_calling", "structured_outputs"],
            "description": "Smaller, faster Grok 3 model",
            "pricing": {"input": 0.30, "output": 0.50},
        },
        "grok-3-fast": {
            "context_length": 131072,
            "capabilities": ["text", "function_calling", "structured_outputs"],
            "description": "Optimized for speed Grok 3 model",
            "pricing": {"input": 5.00, "output": 25.00},
        },
        "grok-3-mini-fast": {
            "context_length": 131072,
            "capabilities": ["text", "function_calling", "structured_outputs"],
            "description": "Fast and efficient mini model",
            "pricing": {"input": 0.60, "output": 4.00},
        },
        "grok-2-vision-1212": {
            "context_length": 32768,
            "capabilities": ["text", "vision", "function_calling"],
            "description": "Grok 2 with vision capabilities",
            "pricing": {"input": 2.00, "output": 10.00},
        },
    }
