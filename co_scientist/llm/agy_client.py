"""Antigravity CLI (agy) provider for AI Co-Scientist.

Translates the project's Anthropic-flavored `AgentCallSpec` into calls routed
through the `google-antigravity` Python SDK (or `agy` CLI binary as fallback),
allowing zero-config execution without manual API keys.
"""

from __future__ import annotations

import json
import os
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
        system_text = "\n".join(b.text for b in spec.system_blocks) if spec.system_blocks else ""
        user_text = "\n".join(b.text for b in spec.user_blocks) if spec.user_blocks else ""

        full_prompt = user_text
        if system_text:
            full_prompt = f"System Instructions:\n{system_text}\n\nUser Request:\n{user_text}"

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
            # Call SDK or CLI fallback
            if self._has_sdk:
                raw_text, stop_reason = await self._call_sdk(full_prompt, effort_level)
            else:
                raw_text, stop_reason = await self._call_cli(full_prompt, effort_level)
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

        # Build adapter message
        msg_id = f"msg_agy_{uuid.uuid4().hex[:12]}"
        blocks = [_Block(type="text", text=raw_text)]
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
        """Execute via agy CLI subprocess fallback."""
        cmd = ["agy", "--print", "--effort", effort, "--prompt", prompt]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        output = proc.stdout or proc.stderr or ""
        return output.strip(), "end_turn"
