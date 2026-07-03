"""
LLM客户端 - 支持多种后端
"""
import os
import asyncio
import random
import aiohttp
from typing import Optional
from pathlib import Path

# Retried on 429 (rate limit) and 5xx (transient server error). NOT retried:
# 4xx other than 429 (bad request / auth — retrying won't help).
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 5
BASE_BACKOFF_S = 1.0
MAX_BACKOFF_S = 30.0

# 自动加载.env文件
def load_env():
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())

load_env()


class LLMClient:
    """统一的LLM客户端接口"""

    def __init__(self, backend: str = "chutes", api_key: Optional[str] = None):
        """
        初始化LLM客户端

        Args:
            backend: "chutes", "ollama", "openai"
            api_key: API密钥(如果需要)
        """
        self.backend = backend
        # 根据backend选择对应的环境变量
        if backend == "openrouter":
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        elif backend == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        else:
            self.api_key = api_key or os.getenv("CHUTES_API_TOKEN")

        # 后端配置
        self.backends = {
            "chutes": {
                "url": "https://llm.chutes.ai/v1/chat/completions",
                "model": "deepseek-ai/DeepSeek-V3",
                "requires_key": True
            },
            "ollama": {
                "url": "http://localhost:11434/api/generate",
                "model": os.getenv("OLLAMA_MODEL", "llama3.2"),  # local, free
                "requires_key": False
            },
            "openai": {
                "url": "https://api.openai.com/v1/chat/completions",
                # "gpt-4" was hardcoded here before -- that's the old, slow,
                # expensive flagship, not a sensible default for a cheap
                # structured-JSON planning task. Override with OPENAI_MODEL
                # if you have access to a newer/cheaper tier (e.g. gpt-5-mini).
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "requires_key": True
            },
            "openrouter": {
                "url": "https://openrouter.ai/api/v1/chat/completions",
                # Override with e.g. OPENROUTER_MODEL="liquid/lfm-2.5-1.2b-instruct:free"
                # for a genuinely free (":free" suffix, no paid credits consumed),
                # cloud-hosted model -- benchmarked ~0.6-2s/call vs ~55-100s/call
                # for local Ollama on this machine's CPU-only hardware.
                "model": os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.2-3b-instruct"),
                "requires_key": True
            }
        }

        if backend not in self.backends:
            raise ValueError(f"Unsupported backend: {backend}")

        self.config = self.backends[backend]

        # 检查API key
        if self.config["requires_key"] and not self.api_key:
            print(f"⚠️  Warning: {backend} requires API key but none provided")

    async def generate(self, prompt: str, temperature: float = 0.3,
                       max_tokens: int = 256) -> str:
        """
        调用LLM生成文本

        Args:
            prompt: 输入prompt
            temperature: 采样温度

        Returns:
            生成的文本
        """
        if self.backend == "ollama":
            return await self._call_ollama(prompt, temperature, max_tokens)
        else:
            return await self._call_openai_compatible(prompt, temperature, max_tokens)

    async def _call_openai_compatible(self, prompt: str, temperature: float,
                                      max_tokens: int = 512) -> str:
        """调用OpenAI兼容的API (Chutes, OpenAI, OpenRouter).

        Retries on 429/5xx with exponential backoff + jitter (honoring a
        Retry-After header when the provider sends one). Before this, any
        429 raised immediately -- under run_fast.py's concurrent batch
        calling, a burst of rate limits meant those trials' plan generation
        permanently failed for that run (see run_fast.py's resume-skip fix:
        a failed call must not be silently lost, and now it has a real
        chance to succeed on retry before falling back to that path).
        """
        headers = {
            "Content-Type": "application/json"
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = {
            "model": self.config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config["url"],
                    headers=headers,
                    json=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]

                    error = await response.text()
                    last_error = f"LLM API error ({response.status}): {error}"

                    if response.status not in RETRYABLE_STATUS or attempt == MAX_RETRIES:
                        raise Exception(last_error)

                    retry_after = response.headers.get("Retry-After")
                    if retry_after is not None:
                        try:
                            delay = float(retry_after)
                        except ValueError:
                            delay = min(BASE_BACKOFF_S * (2 ** attempt), MAX_BACKOFF_S)
                    else:
                        delay = min(BASE_BACKOFF_S * (2 ** attempt), MAX_BACKOFF_S)
                    delay += random.uniform(0, delay * 0.25)  # jitter
                    await asyncio.sleep(delay)

        raise Exception(last_error)

    async def _call_ollama(self, prompt: str, temperature: float,
                           max_tokens: int = 512) -> str:
        """调用本地Ollama"""
        data = {
            "model": self.config["model"],
            "prompt": prompt,
            "stream": False,
            "keep_alive": "30m",          # keep model resident -> no reload per call
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens  # avoid the 128-token default truncating plans
            }
        }

        # No timeout was set here before, so aiohttp's implicit default
        # (300s total) applied silently. On CPU-only local inference with a
        # large RAG-augmented prompt this is easy to exceed, and the
        # resulting asyncio.TimeoutError has an EMPTY str() — it prints as
        # "LLM planning failed: " with nothing after the colon, which looks
        # like a mystery failure rather than what it actually is: a timeout.
        # Set an explicit, generous timeout instead of relying on the
        # invisible default.
        timeout = aiohttp.ClientTimeout(total=900)  # 15 min ceiling for slow local CPU inference
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self.config["url"],
                json=data
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"Ollama error: {error}")

                result = await response.json()
                return result["response"]


# 测试
async def test():
    client = LLMClient(backend="chutes")
    response = await client.generate("Say hello in 3 words")
    print(response)


if __name__ == "__main__":
    asyncio.run(test())
