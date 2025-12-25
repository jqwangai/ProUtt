import json
import os
import random
import asyncio
from typing import Any, Dict, Tuple, Optional

from openai import AsyncOpenAI


JsonDict = Dict[str, Any]
TokenUsage = Dict[str, int]


def build_client(base_url: str, api_key_env: str) -> AsyncOpenAI:
    """
    Create an AsyncOpenAI client using a compatible OpenAI-style endpoint.
    """
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing environment variable: {api_key_env}")
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


async def chat_once_json(
    client: AsyncOpenAI,
    model_id: str,
    system_prompt: str,
    user_content: str,
    retries: int = 3,
    retry_base_sleep_s: float = 1.0,
) -> Tuple[JsonDict, TokenUsage]:
    """
    One JSON-mode chat completion with retry.
    Returns (parsed_json, token_usage).
    """
    last_err: Optional[Exception] = None

    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
            )

            content = (resp.choices[0].message.content or "").strip()
            usage = getattr(resp, "usage", None)
            tokens = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
            }

            parsed = json.loads(content)
            return parsed, tokens

        except json.JSONDecodeError as e:
            last_err = e
        except Exception as e:
            last_err = e

        # Backoff before next attempt
        if attempt < retries - 1:
            await asyncio.sleep(retry_base_sleep_s + random.random())

    raise RuntimeError(f"chat_once_json failed after {retries} retries: {last_err}")
