"""Unit tests for AGYProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from co_scientist.config import Config
from co_scientist.llm.agy_client import AGYProvider, _token_budget_to_effort
from co_scientist.llm.anthropic_client import AgentCallSpec, CachedBlock, CallContext
from co_scientist.llm.budgets import TokenBudget
from co_scientist.llm.provider import get_provider
from co_scientist.llm.routing import ModelRoute


def test_token_budget_to_effort():
    assert _token_budget_to_effort(None) == "low"
    assert _token_budget_to_effort(0) == "low"
    assert _token_budget_to_effort(2000) == "low"
    assert _token_budget_to_effort(6000) == "medium"
    assert _token_budget_to_effort(12000) == "high"


@pytest.mark.asyncio
async def test_get_provider_agy(tmp_cfg: Config, conn):
    budget = TokenBudget(tmp_cfg, 5_000_000, 25.0)
    tmp_cfg.llm.provider = "agy"
    provider = get_provider(tmp_cfg, db=conn, budget=budget)
    assert isinstance(provider, AGYProvider)


@pytest.mark.asyncio
async def test_agy_provider_call(tmp_cfg: Config, conn):
    budget = TokenBudget(tmp_cfg, 5_000_000, 25.0)
    provider = AGYProvider(tmp_cfg, db=conn, budget=budget)

    spec = AgentCallSpec(
        route=ModelRoute("generation", "literature", "gemini-pro"),
        user_blocks=[CachedBlock("Propose a hypothesis for microbiome inflammation.")],
    )
    ctx = CallContext(session_id="test_session", task_id=None, agent="generation", action="generate")

    from datetime import UTC, datetime
    from co_scientist.models import Session, ResearchPlan
    from co_scientist.storage.repos import sessions as sessions_repo

    now = datetime.now(UTC)
    sess = Session(
        id="test_session",
        created_at=now,
        updated_at=now,
        status="running",
        research_goal="test goal",
        research_plan=ResearchPlan(objective="test goal"),
        config_snapshot={},
        budget_tokens=5_000_000,
        budget_usd=25.0,
    )
    await sessions_repo.insert(conn, sess)

    with patch.object(provider, "_call_cli", new_callable=AsyncMock) as mock_cli:
        mock_cli.return_value = ("Test hypothesis: Gut microbes release metabolites.", "end_turn")
        provider._has_sdk = False

        resp = await provider.call(spec, ctx)

        assert resp.raw.content[0].text == "Test hypothesis: Gut microbes release metabolites."
        assert resp.raw.stop_reason == "end_turn"
        assert resp.input_tokens > 0
        assert resp.output_tokens > 0


@pytest.mark.asyncio
async def test_agy_provider_high_budget_pro_tier(tmp_cfg: Config, conn):
    """Test AGYProvider behavior when user has high token budget (Google Pro limits).
    Verifies that high thinking budgets map to 'medium' and 'high' reasoning effort levels.
    """
    tmp_cfg.thinking.generation_literature = 16000  # High budget
    tmp_cfg.thinking.ranking_debate = 8000         # Medium budget

    budget = TokenBudget(tmp_cfg, 20_000_000, 100.0)
    provider = AGYProvider(tmp_cfg, db=conn, budget=budget)

    from datetime import UTC, datetime
    from co_scientist.models import Session, ResearchPlan
    from co_scientist.storage.repos import sessions as sessions_repo

    now = datetime.now(UTC)
    sess = Session(
        id="pro_session",
        created_at=now,
        updated_at=now,
        status="running",
        research_goal="pro test goal",
        research_plan=ResearchPlan(objective="pro test goal"),
        config_snapshot={},
        budget_tokens=20_000_000,
        budget_usd=100.0,
    )
    await sessions_repo.insert(conn, sess)

    # 1. Test High Effort mapping for literature generation (16k budget)
    spec_high = AgentCallSpec(
        route=ModelRoute("generation", "literature", "gemini-pro"),
        user_blocks=[CachedBlock("Deep scientific hypothesis generation")],
    )
    ctx_high = CallContext(session_id="pro_session", task_id=None, agent="generation", action="generate", mode="literature")

    with patch.object(provider, "_call_cli", new_callable=AsyncMock) as mock_cli:
        mock_cli.return_value = ("High-reasoning hypothesis proposal.", "end_turn")
        provider._has_sdk = False

        await provider.call(spec_high, ctx_high)
        # Assert that 'high' effort was passed to CLI call
        mock_cli.assert_called_once_with(
            "User Request:\nDeep scientific hypothesis generation", "high"
        )

    # 2. Test Medium Effort mapping for ranking debate (8k budget)
    spec_med = AgentCallSpec(
        route=ModelRoute("ranking", "debate", "gemini-pro"),
        user_blocks=[CachedBlock("Debate two competing hypotheses")],
    )
    ctx_med = CallContext(session_id="pro_session", task_id=None, agent="ranking", action="debate", mode="debate")

    with patch.object(provider, "_call_cli", new_callable=AsyncMock) as mock_cli:
        mock_cli.return_value = ("Medium-reasoning debate judgment.", "end_turn")
        provider._has_sdk = False

        await provider.call(spec_med, ctx_med)
        # Assert that 'medium' effort was passed to CLI call
        mock_cli.assert_called_once_with(
            "User Request:\nDebate two competing hypotheses", "medium"
        )


@pytest.mark.asyncio
async def test_agy_provider_tool_parsing(tmp_cfg: Config, conn):
    """Test that AGYProvider correctly parses tool call JSON blocks from raw LLM output."""
    budget = TokenBudget(tmp_cfg, 1_000_000, 10.0)
    provider = AGYProvider(tmp_cfg, db=conn, budget=budget)

    from datetime import UTC, datetime
    from co_scientist.models import Session, ResearchPlan
    from co_scientist.storage.repos import sessions as sessions_repo

    now = datetime.now(UTC)
    sess = Session(
        id="tool_session",
        created_at=now,
        updated_at=now,
        status="running",
        research_goal="tool test goal",
        research_plan=ResearchPlan(objective="tool test goal"),
        config_snapshot={},
        budget_tokens=1_000_000,
        budget_usd=10.0,
    )
    await sessions_repo.insert(conn, sess)

    spec = AgentCallSpec(
        route=ModelRoute("generation", "literature", "gemini-pro"),
        user_blocks=[CachedBlock("Generate a hypothesis")],
        tools=[{"name": "record_hypothesis", "description": "Record hypothesis", "input_schema": {}}],
        tool_choice={"type": "tool", "name": "record_hypothesis"},
    )
    ctx = CallContext(session_id="tool_session", task_id=None, agent="generation", action="generate", mode="literature")

    raw_llm_json = 'Here is my hypothesis:\n```json\n{\n  "statement": "Test statement",\n  "mechanism": "Test mechanism"\n}\n```'

    with patch.object(provider, "_call_cli", new_callable=AsyncMock) as mock_cli:
        mock_cli.return_value = (raw_llm_json, "end_turn")
        provider._has_sdk = False

        resp = await provider.call(spec, ctx)
        assert resp.raw.stop_reason == "tool_use"
        tool_blocks = [b for b in resp.raw.content if b.type == "tool_use"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0].name == "record_hypothesis"
        assert tool_blocks[0].input["statement"] == "Test statement"


@pytest.mark.asyncio
async def test_agy_provider_markdown_fallback(tmp_cfg: Config, conn):
    """Test that AGYProvider extracts structured tool calls even from raw markdown responses."""
    budget = TokenBudget(tmp_cfg, 1_000_000, 10.0)
    provider = AGYProvider(tmp_cfg, db=conn, budget=budget)

    from datetime import UTC, datetime
    from co_scientist.models import Session, ResearchPlan
    from co_scientist.storage.repos import sessions as sessions_repo

    now = datetime.now(UTC)
    sess = Session(
        id="markdown_session",
        created_at=now,
        updated_at=now,
        status="running",
        research_goal="markdown test goal",
        research_plan=ResearchPlan(objective="markdown test goal"),
        config_snapshot={},
        budget_tokens=1_000_000,
        budget_usd=10.0,
    )
    await sessions_repo.insert(conn, sess)

    spec = AgentCallSpec(
        route=ModelRoute("reflection", "full", "gemini-pro"),
        user_blocks=[CachedBlock("Review this hypothesis")],
        tools=[{"name": "record_review", "description": "Record review", "input_schema": {}}],
        tool_choice={"type": "tool", "name": "record_review"},
    )
    ctx = CallContext(session_id="markdown_session", task_id=None, agent="reflection", action="review", mode="full")

    raw_markdown = "# Critical Review\n\n### 2. Novelty\nScore: 0.75\n\n### 3. Correctness\nScore: 0.65\n\n### 5. Verdict\nVerdict: missing_piece"

    with patch.object(provider, "_call_cli", new_callable=AsyncMock) as mock_cli:
        mock_cli.return_value = (raw_markdown, "end_turn")
        provider._has_sdk = False

        resp = await provider.call(spec, ctx)
        assert resp.raw.stop_reason == "tool_use"
        tool_blocks = [b for b in resp.raw.content if b.type == "tool_use"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0].name == "record_review"
        assert tool_blocks[0].input["verdict"] == "missing_piece"
        assert tool_blocks[0].input["novelty"] == 0.75


