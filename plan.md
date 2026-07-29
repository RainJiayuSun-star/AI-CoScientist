# Architecture & Implementation Plan: Antigravity CLI (`agy`) Integration for AI Co-Scientist

## 1. Overview & Objective

This document details the architectural decisions and implementation plan for integrating **Antigravity CLI (`agy`)** into the `AI-CoScientist` multi-agent framework.

The primary goal is to allow `co-scientist` to run using local `agy` authentication and model infrastructure as a zero-config, optional LLM provider backend. This eliminates the requirement for manual vendor API key configuration (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) while keeping all existing direct API provider options fully supported.

---

## 2. Key Architecture Decisions

### 2.1 Provider Abstraction & `AGYProvider`
* **Interface Protocol:** Implement a new `AGYProvider` class under `co_scientist/llm/agy_client.py` adhering to the `LLMProvider` protocol defined in `co_scientist/llm/provider.py`.
* **Execution Path:**
  1. **Primary Route (`google-antigravity` Python SDK):** Leverages `google.antigravity.Agent` async context manager for stream handling, local session authentication, and tool execution.
  2. **Fallback Route (`agy` CLI Binary Subprocess):** Invokes `agy` CLI in non-interactive print mode (`agy --print --output-format json`) if the Python SDK package is not installed.
* **Normalized Response Translation:** Translates `agy` outputs into standard `AnthropicResponse` format (`.content`, `.stop_reason`, `.usage`) expected by all `co-scientist` downstream agents.

### 2.2 Tool Execution & Schema Mapping
`co-scientist` heavily relies on tool calling (e.g., `web_fetch`, `pubmed_search`). The integration must map these schemas accurately:
* **Schema Translation:** `AGYProvider` will dynamically translate `AgentCallSpec.tools` (JSON schemas) into the `google.antigravity` custom tool registry format so the AGY model understands them.
* **Response Normalization:** When the SDK yields a `ToolCall` event, `AGYProvider` will format it into an Anthropic `ToolUseBlock`, yield control back to the `co-scientist` tool loop, and pass the result back as a `ToolResultBlock`.

### 2.3 Extended Reasoning (`thinking` budget mapping)
`co-scientist` allocates token budgets for complex reasoning tasks (e.g., 8000 tokens for generation).
* `AGYProvider` will map these numerical token limits to `agy`'s `effort` levels (`low`, `medium`, `high`) automatically.
  * Budget < 2000 tokens ➔ `effort: low`
  * Budget 2000 - 8000 tokens ➔ `effort: medium`
  * Budget > 8000 tokens ➔ `effort: high`

### 2.4 Two-Tier Model Strategy
`co-scientist` delegates tasks across 11 model configuration slots in `[models]`. We map these to `agy` model tiers:

* **⚡ Tier 1: Fast / High-Throughput Tier (`gemini-flash`)**
  * `parse_goal`, `ranking_pairwise`, `metareview_feedback`, `classifier`, `judge`

* **🧠 Tier 2: Strong Reasoning & Evaluation Tier (`gemini-pro`)**
  * `generation`, `reflection`, `evolution`, `ranking_debate`, `ranking_priority`, `metareview_final`

```toml
[llm]
provider = "agy"

[models]
parse_goal          = "gemini-flash"
generation          = "gemini-pro"
reflection          = "gemini-pro"
evolution           = "gemini-pro"
ranking_pairwise    = "gemini-flash"
ranking_debate      = "gemini-pro"
ranking_priority    = "gemini-pro"
metareview_feedback = "gemini-flash"
metareview_final    = "gemini-pro"
classifier          = "gemini-flash"
judge               = "gemini-flash"
```

### 2.5 Periodic Reporting & Human-in-the-Loop Steering
* **Real-time Web UI (`co-scientist serve`):** Exposes a FastAPI dashboard for watching hypothesis generation, live review transcripts, and Elo rankings.
* **Periodic CLI Reporting (`co-scientist status <id>`):** Provides periodic progress updates on active tasks and token usage.
* **Mid-Run Steering (`co-scientist feedback <id>`):** Allows injecting human guidance mid-session.

---

## 3. Implementation Roadmap

### Phase 1: Python Dependencies
- [ ] Add `google-antigravity` as an optional dependency in `pyproject.toml` (`[project.optional-dependencies] agy = ["google-antigravity>=0.1.0"]`).

### Phase 2: AGY LLM Provider Client (`co_scientist/llm/agy_client.py`)
- [ ] Create `AGYProvider` class implementing `LLMProvider` interface.
- [ ] Translate `AgentCallSpec` (messages, system instructions) into `google.antigravity.Agent` context.
- [ ] Implement Tool Schema Translation (convert `AgentCallSpec.tools` for AGY SDK).
- [ ] Implement Effort Mapping (convert `[thinking]` token limits into `low/medium/high` effort).
- [ ] Parse `agy` outputs and tool calls into normalized `AnthropicResponse` objects.

### Phase 3: Registry & Configuration Updates
- [ ] Add `"agy"` and `"antigravity"` to `KNOWN_PROVIDERS` in `co_scientist/llm/provider.py`.
- [ ] Update `get_provider()` factory method to instantiate `AGYProvider`.
- [ ] Add `agy` provider section and model mapping to `config/default.toml`.

### Phase 4: CLI Shortcuts & Auto-Detection
- [ ] Add `--use-agy` flag to `co-scientist run` and `co-scientist init` in `co_scientist/cli.py`.
- [ ] Skip API key validation warnings when `--use-agy` or `provider = "agy"` is active.

### Phase 5: Verification & Testing
- [ ] Create `tests/test_agy_provider.py` with mock SDK & tool responses.
- [ ] Verify `co-scientist init --use-agy` initializes cleanly without manual API keys in `.env`.
- [ ] Run a test session (`co-scientist run "Identify novel therapeutic targets for AML" --use-agy`).
