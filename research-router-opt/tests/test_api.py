import asyncio
from pathlib import Path

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response

from research_router_opt.api import create_app
from research_router_opt.data import build_datasets
from research_router_opt.train import train_policy


async def post(app: FastAPI, query: str) -> Response:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/route", json={"query": query})


def test_route_uses_saved_policy(tmp_path: Path) -> None:
    training_tasks, _ = build_datasets()
    policy_path = tmp_path / "policy.json"
    train_policy(training_tasks).save(policy_path)
    response = asyncio.run(post(create_app(policy_path), "总结这篇亚洲象论文的摘要"))

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_tool"] == "abstract_summarize"
    assert payload["reward_unavailable"] is True


def test_route_returns_503_without_policy(tmp_path: Path) -> None:
    response = asyncio.run(post(create_app(tmp_path / "missing.json"), "检索雪豹论文"))
    assert response.status_code == 503
