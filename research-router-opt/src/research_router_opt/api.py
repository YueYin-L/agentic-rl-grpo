"""FastAPI demo for the trained routing policy."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from research_router_opt.agent import ResearchRouterAgent
from research_router_opt.models import RoutingTask
from research_router_opt.policy import LinUCBPolicy


class RouteRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)


class RouteResponse(BaseModel):
    selected_tool: str
    reward_unavailable: bool = True
    observation: dict[str, object]


def create_app(policy_path: Path) -> FastAPI:
    application = FastAPI(title="Research Router Optimization", version="0.1.0")

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post("/route", response_model=RouteResponse)
    def route(request: RouteRequest) -> RouteResponse:
        if not policy_path.exists():
            raise HTTPException(status_code=503, detail="Trained policy artifact is unavailable.")
        policy = LinUCBPolicy.load(policy_path)
        task = RoutingTask(
            task_id=f"api-{uuid4().hex}",
            query=request.query,
            # Online requests have no ground-truth label; use the selected action only to execute.
            expected_tool=policy.select_tool(request.query),
        )
        trajectory = ResearchRouterAgent(policy).run(task)
        step = trajectory.steps[0]
        return RouteResponse(
            selected_tool=step.action,
            observation={
                "tool": step.tool_output.tool,
                "status": step.tool_output.status,
                "result": step.tool_output.result,
            },
        )

    return application


DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "policy.json"
app = create_app(DEFAULT_POLICY_PATH)

