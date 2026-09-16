from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional
from urllib.request import Request, urlopen


class LLMProvider:
    """Small provider interface used only by the RAG layer."""

    def chat(self, model: str, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    def __init__(self) -> None:
        self._module = None

    def chat(self, model: str, messages: list[dict[str, str]]) -> str:
        if self._module is None:
            try:
                import ollama  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "Ollama provider selected, but the 'ollama' Python package is not installed. "
                    "Install Protein_RAG/requirements-rag.txt or select another provider."
                ) from exc
            self._module = ollama

        response = self._module.chat(model=model, messages=messages, stream=False, think=False)
        content = (getattr(response.message, "content", "") or "").strip()
        if not content:
            raise RuntimeError("Ollama returned an empty final answer.")
        return content


class OpenAICompatibleProvider(LLMProvider):
    """Calls any OpenAI-compatible /v1/chat/completions endpoint via stdlib only."""

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self.base_url = (base_url or os.getenv("RAG_LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("RAG_LLM_API_KEY", "")
        if not self.base_url:
            raise RuntimeError("RAG_LLM_BASE_URL is required for the openai_compatible provider.")
        if not self.api_key:
            raise RuntimeError("RAG_LLM_API_KEY is required for the openai_compatible provider.")

    def chat(self, model: str, messages: list[dict[str, str]]) -> str:
        url = self.base_url if self.base_url.endswith("/chat/completions") else f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=float(os.getenv("RAG_LLM_TIMEOUT", "90"))) as response:
                body = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"OpenAI-compatible LLM request failed: {exc}") from exc
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("OpenAI-compatible endpoint returned an unexpected response format.") from exc
        content = str(content or "").strip()
        if not content:
            raise RuntimeError("OpenAI-compatible endpoint returned an empty final answer.")
        return content


def build_llm_provider() -> LLMProvider:
    provider = os.getenv("RAG_LLM_PROVIDER", "ollama").strip().lower()
    if provider == "ollama":
        return OllamaProvider()
    if provider in {"openai_compatible", "openai-compatible", "api"}:
        return OpenAICompatibleProvider()
    raise RuntimeError(f"Unsupported RAG_LLM_PROVIDER={provider!r}. Use 'ollama' or 'openai_compatible'.")
