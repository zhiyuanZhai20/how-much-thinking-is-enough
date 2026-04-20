"""Unified API client wrapper with retry + concurrency control.

All providers we use (DeepSeek, DashScope/Qwen, OpenAI) speak the OpenAI Chat
Completions schema, so a single OpenAI-SDK client per base_url is enough.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import config


@dataclass
class ChatResult:
    content: str
    reasoning: str  # may be ""
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int  # 0 if not reported separately


def _get_client(provider: str, async_client: bool = True):
    if provider == "deepseek":
        kwargs = dict(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)
    elif provider == "dashscope":
        kwargs = dict(api_key=config.DASHSCOPE_API_KEY, base_url=config.DASHSCOPE_BASE_URL)
    elif provider == "openai":
        kwargs = dict(api_key=config.OPENAI_API_KEY)
    else:
        raise ValueError(f"unknown provider {provider}")
    return AsyncOpenAI(**kwargs) if async_client else OpenAI(**kwargs)


_client_cache = {}

def get_client(provider: str, async_client: bool = True):
    key = (provider, async_client)
    if key not in _client_cache:
        _client_cache[key] = _get_client(provider, async_client)
    return _client_cache[key]


@retry(
    stop=stop_after_attempt(config.RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
async def chat_async(
    provider: str,
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 4096,
    extra_body: Optional[dict] = None,
) -> ChatResult:
    client = get_client(provider, async_client=True)
    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if extra_body:
        kwargs["extra_body"] = extra_body

    # DashScope reasoning models require streaming when reasoning is on.
    needs_stream = provider == "dashscope" and (
        "qwq" in model or "r1-distill" in model or "thinking" in model
    )
    if needs_stream:
        kwargs["stream"] = True
        kwargs["extra_body"] = {**(extra_body or {}), "enable_thinking": True}
        return await _stream_chat(client, **kwargs)

    resp = await client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    content = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", "") or ""
    usage = resp.usage
    p_tok = usage.prompt_tokens if usage else 0
    c_tok = usage.completion_tokens if usage else 0
    r_tok = 0
    if usage and hasattr(usage, "completion_tokens_details"):
        details = usage.completion_tokens_details
        if details and hasattr(details, "reasoning_tokens"):
            r_tok = details.reasoning_tokens or 0
    return ChatResult(content=content, reasoning=reasoning,
                      prompt_tokens=p_tok, completion_tokens=c_tok,
                      reasoning_tokens=r_tok)


async def _stream_chat(client, **kwargs) -> ChatResult:
    content_parts = []
    reasoning_parts = []
    p_tok = c_tok = r_tok = 0
    stream = await client.chat.completions.create(**kwargs)
    async for chunk in stream:
        if not chunk.choices:
            if getattr(chunk, "usage", None):
                u = chunk.usage
                p_tok = u.prompt_tokens or 0
                c_tok = u.completion_tokens or 0
            continue
        delta = chunk.choices[0].delta
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            reasoning_parts.append(delta.reasoning_content)
        if delta.content:
            content_parts.append(delta.content)
        if getattr(chunk, "usage", None):
            u = chunk.usage
            p_tok = u.prompt_tokens or p_tok
            c_tok = u.completion_tokens or c_tok
    return ChatResult(
        content="".join(content_parts),
        reasoning="".join(reasoning_parts),
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        reasoning_tokens=r_tok,
    )


class ConcurrencyLimiter:
    def __init__(self, n: int):
        self.sem = asyncio.Semaphore(n)
    async def __aenter__(self):
        await self.sem.acquire()
    async def __aexit__(self, *a):
        self.sem.release()
