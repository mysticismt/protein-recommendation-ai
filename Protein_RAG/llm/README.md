# RAG LLM Providers

The RAG layer supports two generation modes without changing the legacy ML project:

- `ollama` (optional): local model runtime such as the existing Qwen setup.
- `openai_compatible`: any hosted endpoint that implements `/v1/chat/completions`.

Hosted mode uses:

```text
RAG_LLM_PROVIDER=openai_compatible
RAG_LLM_BASE_URL=https://YOUR_PROVIDER.example/v1
RAG_LLM_API_KEY=YOUR_TOKEN
RAG_LLM_MODEL=YOUR_MODEL_NAME
```

The default API and pipeline path does **not** invoke an LLM. Use `generate=true` only when a provider has been configured. When `generate=false` is sent to the API, the LLM generation step is skipped entirely. This allows the ML + SHAP + retrieval + deterministic contract to run without Ollama, Qwen, or any other model installation.
