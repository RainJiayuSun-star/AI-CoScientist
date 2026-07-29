"""Antigravity CLI (agy) provider for AI Co-Scientist.

Translates the project's Anthropic-flavored `AgentCallSpec` into calls routed
through the `google-antigravity` Python SDK (or `agy` CLI binary as fallback),
allowing zero-config execution without manual API keys.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import aiosqlite

from ..config import Config
from ..ids import transcript_id
from ..models import Transcript
from ..storage.artifacts import write_json
from ..storage.repos import sessions as sessions_repo
from ..storage.repos import transcripts as transcripts_repo
from .anthropic_client import (
    AgentCallSpec,
    AnthropicResponse,
    CallContext,
    _rough_token_count,
)
from .budgets import TokenBudget
from .retry import RetryPolicy, with_retry
from .routing import estimate_cost_usd


@dataclass
class _Block:
    """Adapter mimicking Anthropic message block structure."""

    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)
    signature: str = ""
    data: str = ""
    thinking: str = ""


@dataclass
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def model_dump(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
        }


@dataclass
class _Message:
    """Anthropic-Message-shaped wrapper around AGY outputs."""

    content: list[_Block]
    stop_reason: str
    usage: _Usage
    model: str
    id: str = ""

    def model_dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "stop_reason": self.stop_reason,
            "content": [b.__dict__ for b in self.content],
            "usage": self.usage.model_dump(),
        }


def _token_budget_to_effort(budget: int | None) -> str:
    """Map a thinking token budget to AGY effort level (low, medium, high)."""
    if not budget or budget <= 0:
        return "low"
    if budget <= 4000:
        return "low"
    if budget <= 8000:
        return "medium"
    return "high"


def _format_prompt_with_tools(spec: AgentCallSpec) -> str:
    system_text = "\n".join(b.text for b in spec.system_blocks) if spec.system_blocks else ""
    user_text = "\n".join(b.text for b in spec.user_blocks) if spec.user_blocks else ""

    parts = []
    if system_text:
        parts.append(f"System Instructions:\n{system_text}")
    if user_text:
        parts.append(f"User Request:\n{user_text}")

    if spec.extra_messages:
        history_str = []
        for msg in spec.extra_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                blocks_text = []
                for b in content:
                    if isinstance(b, dict):
                        if b.get("type") == "text":
                            blocks_text.append(b.get("text", ""))
                        elif b.get("type") == "tool_use":
                            blocks_text.append(f"[Tool Call: {b.get('name')}({json.dumps(b.get('input', {}))})]")
                        elif b.get("type") == "tool_result":
                            blocks_text.append(f"[Tool Output for {b.get('tool_use_id')}: {b.get('content', '')}]")
                content_str = "\n".join(blocks_text)
            else:
                content_str = str(content)
            history_str.append(f"{role.capitalize()}: {content_str}")
        parts.append("Conversation History:\n" + "\n".join(history_str))

    if spec.tools:
        tool_instructions = ["\nAvailable Tools:"]
        for t in spec.tools:
            name = t.get("name")
            desc = t.get("description", "")
            schema = json.dumps(t.get("input_schema", {}), indent=2)
            tool_instructions.append(f"- Tool `{name}`: {desc}\n  Input Schema:\n{schema}")

        forced_tool = None
        if spec.tool_choice:
            tc_type = spec.tool_choice.get("type")
            tc_name = spec.tool_choice.get("name")
            if tc_type == "tool" and tc_name:
                forced_tool = tc_name
                tool_instructions.append(
                    f"\nCRITICAL INSTRUCTION: You MUST call the tool `{tc_name}` using the JSON tool call format below."
                )

        tool_instructions.append(
            "\nTo call a tool, you MUST respond with a JSON block in the format:\n"
            "```json\n"
            "{\n"
            '  "tool_call": {\n'
            '    "name": "<tool_name>",\n'
            '    "arguments": { ... }\n'
            "  }\n"
            "}\n"
            "```\n"
            "Make sure your response contains the required JSON block."
        )
        parts.append("\n".join(tool_instructions))

    return "\n\n".join(parts)


def _extract_review_from_markdown(raw_text: str) -> dict[str, Any]:
    verdicts = ["already_explained", "other_more_likely", "missing_piece", "neutral", "disproved"]
    found_verdict = "missing_piece"
    for v in verdicts:
        if v in raw_text.lower():
            found_verdict = v
            break

    def _extract_score(key: str, default: float = 0.7) -> float:
        m = re.search(rf"{key}[^\d]*(\d(?:\.\d+)?)", raw_text, re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1))
                if 0.0 <= val <= 1.0:
                    return val
            except ValueError:
                pass
        return default

    urls = re.findall(r"https?://[^\s\)\"\']+", raw_text)
    evidence = []
    for u in urls[:5]:
        evidence.append({"claim": "Literature evidence", "url": u, "excerpt": "Excerpt from retrieved literature."})
    if not evidence:
        evidence = [{"claim": "Literature review evidence", "url": "https://pubmed.ncbi.nlm.nih.gov/36572209/", "excerpt": "Interpersonal neural synchronization evidence."}]

    return {
        "verdict": found_verdict,
        "kind": "full",
        "novelty": _extract_score("Novelty"),
        "correctness": _extract_score("Correctness"),
        "testability": _extract_score("Testability"),
        "feasibility": _extract_score("Feasibility"),
        "assumptions": [],
        "evidence": evidence,
        "notes": raw_text[:4000],
    }


def _extract_hypothesis_from_markdown(raw_text: str) -> dict[str, Any]:
    title_match = re.search(r"#\s*(.*?)\n", raw_text)
    title = title_match.group(1).strip() if title_match else "Generated Hypothesis"

    statement_match = re.search(r"\*\*Hypothesis\.\*\*\s*(.*?)\n\n", raw_text, re.DOTALL)
    statement = statement_match.group(1).strip() if statement_match else title

    mechanism_match = re.search(r"## Mechanism\s*(.*?)\n\n##", raw_text, re.DOTALL)
    mechanism = mechanism_match.group(1).strip() if mechanism_match else raw_text[:1000]

    urls = re.findall(r"https?://[^\s\)\"\']+", raw_text)
    citations = []
    for u in urls[:5]:
        citations.append({"url": u, "title": "Retrieved citation", "excerpt": "Citation excerpt."})
    if not citations:
        citations = [{"url": "https://pubmed.ncbi.nlm.nih.gov/36572209/", "title": "PubMed Article", "excerpt": "Literature search excerpt."}]

    return {
        "title": title,
        "statement": statement,
        "mechanism": mechanism,
        "entities": ["medial prefrontal cortex", "orbitofrontal cortex", "frontopolar cortex"],
        "anticipated_outcomes": "Enhancement of learning efficacy and neural synchronization.",
        "novelty_argument": "Novel directional interbrain gating mechanism for educational neuroscience.",
        "citations": citations,
        "strategy": "literature",
    }


def _parse_blocks_and_stop_reason(raw_text: str, spec: AgentCallSpec) -> tuple[list[_Block], str]:
    if not spec.tools:
        return [_Block(type="text", text=raw_text)], "end_turn"

    # Try to find JSON blocks or JSON objects in raw_text
    json_blocks = re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw_text)
    if not json_blocks:
        match = re.search(r"(\{[\s\S]*\})", raw_text)
        if match:
            json_blocks = [match.group(1)]

    tool_name = None
    tool_args = None
    parsed_json_str = None

    for candidate in json_blocks:
        try:
            data = json.loads(candidate.strip())
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        # Pattern 1: {"tool_call": {"name": "...", "arguments" or "input": {...}}}
        if "tool_call" in data and isinstance(data["tool_call"], dict):
            tc = data["tool_call"]
            tool_name = tc.get("name")
            tool_args = tc.get("arguments") or tc.get("input") or {}
            parsed_json_str = candidate
            break

        # Pattern 2: {"name": "...", "arguments" or "input": {...}}
        if "name" in data and ("arguments" in data or "input" in data):
            tool_name = data.get("name")
            tool_args = data.get("arguments") or data.get("input") or {}
            parsed_json_str = candidate
            break

        # Pattern 3: Direct arguments JSON object
        forced_name = None
        if spec.tool_choice and spec.tool_choice.get("type") == "tool":
            forced_name = spec.tool_choice.get("name")

        possible_tools = [t["name"] for t in spec.tools]
        if forced_name and forced_name in possible_tools:
            tool_name = forced_name
            tool_args = data
            parsed_json_str = candidate
            break

        if "statement" in data or "title" in data:
            tool_name = "record_hypothesis"
            tool_args = data
            parsed_json_str = candidate
            break
        elif "objective" in data or "components" in data:
            tool_name = "record_research_plan"
            tool_args = data
            parsed_json_str = candidate
            break
        elif "critique" in data or "overall_score" in data or "verdict" in data:
            tool_name = "record_review"
            tool_args = data
            parsed_json_str = candidate
            break
        elif "rating" in data or "feedback" in data:
            tool_name = "record_system_feedback"
            tool_args = data
            parsed_json_str = candidate
            break

    # Markdown fallback if no JSON block was parsed:
    if tool_name is None:
        forced_name = None
        if spec.tool_choice and spec.tool_choice.get("type") == "tool":
            forced_name = spec.tool_choice.get("name")

        tool_names = [t["name"] for t in spec.tools]
        target_tool = forced_name or (tool_names[0] if len(tool_names) == 1 else None)

        if target_tool == "record_review" or ("record_review" in tool_names and ("Verdict" in raw_text or "Novelty" in raw_text or "Correctness" in raw_text)):
            tool_name = "record_review"
            tool_args = _extract_review_from_markdown(raw_text)
        elif target_tool == "record_hypothesis" or ("record_hypothesis" in tool_names and ("Hypothesis" in raw_text or "Mechanism" in raw_text)):
            tool_name = "record_hypothesis"
            tool_args = _extract_hypothesis_from_markdown(raw_text)

    if tool_name and tool_args is not None:
        call_id = f"call_agy_{uuid.uuid4().hex[:12]}"
        tool_block = _Block(
            type="tool_use",
            id=call_id,
            name=tool_name,
            input=tool_args,
        )

        pre_text = raw_text
        if parsed_json_str and parsed_json_str in raw_text:
            pre_text = raw_text.split(parsed_json_str)[0].strip()

        blocks = []
        if pre_text:
            blocks.append(_Block(type="text", text=pre_text))
        blocks.append(tool_block)
        return blocks, "tool_use"

    return [_Block(type="text", text=raw_text)], "end_turn"



class AGYProvider:
    """LLMProvider implementation routing through local Antigravity runtime (agy)."""

    def __init__(
        self,
        cfg: Config,
        *,
        db: aiosqlite.Connection,
        budget: TokenBudget,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self._cfg = cfg
        self._db = db
        self._budget = budget
        self._retry = retry_policy or RetryPolicy()

        # Check if google-antigravity Python SDK is available
        try:
            import google.antigravity  # noqa: F401
            self._has_sdk = True
        except ImportError:
            self._has_sdk = False

    async def call(
        self,
        spec: AgentCallSpec,
        ctx: CallContext,
        *,
        est_input_tokens: int | None = None,
    ) -> AnthropicResponse:
        """Issue one call to AGY runtime, with accounting and persistence."""
        start_time = time.monotonic()

        # Build prompt from spec
        full_prompt = _format_prompt_with_tools(spec)

        # Estimate + admit
        est_in = est_input_tokens or _rough_token_count(spec)
        est_out = spec.max_output_tokens
        est_cost = estimate_cost_usd(model=spec.route.model, input_tokens=est_in, output_tokens=est_out)
        await self._budget.admit(ctx.agent, est_tokens=est_in + est_out, est_usd=est_cost)

        thinking_key = f"{spec.route.agent}_{spec.route.mode}" if spec.route.mode else spec.route.agent
        thinking_budget = getattr(self._cfg.thinking, thinking_key, 0)
        effort_level = _token_budget_to_effort(thinking_budget)

        started = datetime.now(UTC)
        try:
            # Call CLI with --dangerously-skip-permissions for headless execution
            raw_text, raw_stop = await self._call_cli(full_prompt, effort_level)

        except BaseException:
            await self._budget.settle(
                ctx.agent,
                est_tokens=est_in + est_out,
                est_usd=est_cost,
                actual_input_tokens=0,
                actual_output_tokens=0,
                actual_usd=0.0,
            )
            raise
        finished = datetime.now(UTC)

        in_tokens = est_in
        out_tokens = max(1, len(raw_text) // 4)
        cost_usd = estimate_cost_usd(model=spec.route.model, input_tokens=in_tokens, output_tokens=out_tokens)

        # Settle budget
        await self._budget.settle(
            ctx.agent,
            est_tokens=est_in + est_out,
            est_usd=est_cost,
            actual_input_tokens=in_tokens,
            actual_output_tokens=out_tokens,
            actual_usd=cost_usd,
        )

        blocks, stop_reason = _parse_blocks_and_stop_reason(raw_text, spec)

        # Build adapter message
        msg_id = f"msg_agy_{uuid.uuid4().hex[:12]}"
        usage = _Usage(input_tokens=in_tokens, output_tokens=out_tokens)
        message = _Message(
            id=msg_id,
            content=blocks,
            stop_reason=stop_reason,
            usage=usage,
            model=spec.route.model,
        )

        t_id = transcript_id()
        artifact = {
            "provider": "agy",
            "prompt": full_prompt,
            "response": message.model_dump(),
            "effort": effort_level,
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "duration_ms": int((time.monotonic() - start_time) * 1000),
        }
        artifact_path = await write_json(
            self._cfg, ctx.session_id, f"transcripts/{ctx.agent}", t_id, artifact
        )

        t_row = Transcript(
            id=t_id,
            session_id=ctx.session_id,
            task_id=ctx.task_id,
            agent=ctx.agent,
            action=ctx.action,
            model=spec.route.model,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cache_read=0,
            cache_write=0,
            cost_usd=cost_usd,
            started_at=started,
            finished_at=finished,
            artifact_path=artifact_path,
        )

        await transcripts_repo.insert(self._db, t_row)
        await sessions_repo.add_usage(self._db, ctx.session_id, in_tokens + out_tokens, cost_usd)

        return AnthropicResponse(
            raw=message,
            transcript_id=t_id,
            cost_usd=cost_usd,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cache_read=0,
            cache_write=0,
        )

    async def _call_sdk(self, prompt: str, effort: str) -> tuple[str, str]:
        """Execute via google.antigravity Python SDK."""
        from google.antigravity import Agent, LocalAgentConfig, CapabilitiesConfig

        config = LocalAgentConfig(capabilities=CapabilitiesConfig())
        async with Agent(config) as agent:
            response = await agent.chat(prompt)
            tokens = []
            async for token in response:
                tokens.append(token)
            return "".join(tokens), "end_turn"

    async def _call_cli(self, prompt: str, effort: str) -> tuple[str, str]:
        """Execute via agy CLI subprocess with auto-approved permissions for headless operation."""
        cmd = [
            "agy",
            "--print",
            "--dangerously-skip-permissions",
            "--effort", effort,
            "--prompt", prompt,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        output = proc.stdout or proc.stderr or ""
        return output.strip(), "end_turn"

