"""Baseline and LinUCB policies for bounded tool selection."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Protocol

from research_router_opt.features import FEATURE_NAMES, extract_features
from research_router_opt.models import TOOL_NAMES


class RoutingPolicy(Protocol):
    name: str

    def select_tool(self, query: str) -> str:
        ...


class KeywordBaselinePolicy:
    """Intentionally simple priority router used as a reproducible baseline."""

    name = "keyword_baseline"

    _RULES = (
        ("species_lookup", ("物种", "分布", "学名", "栖息地", "保护级别")),
        ("paper_search", ("论文", "文献", "检索", "搜索", "发表")),
        ("evidence_compare", ("比较", "对比", "差异", "优缺点")),
        ("abstract_summarize", ("总结", "概括", "摘要", "提炼", "归纳")),
    )

    def select_tool(self, query: str) -> str:
        normalized = query.strip().lower()
        if not normalized:
            raise ValueError("Query must not be empty.")
        for tool, terms in self._RULES:
            if any(term in normalized for term in terms):
                return tool
        return "paper_search"


def _identity(size: int) -> list[list[float]]:
    return [[1.0 if row == column else 0.0 for column in range(size)] for row in range(size)]


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve Ax=b with partial-pivot Gaussian elimination."""

    size = len(vector)
    augmented = [matrix[row][:] + [vector[row]] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("Policy matrix is singular.")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                current - factor * pivot_current
                for current, pivot_current in zip(augmented[row], augmented[column], strict=True)
            ]
    return [augmented[row][-1] for row in range(size)]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


class LinUCBPolicy:
    """Per-action linear contextual bandit with upper-confidence exploration."""

    name = "linucb_router"

    def __init__(self, alpha: float = 0.8) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = alpha
        self.dimension = len(FEATURE_NAMES)
        self.actions = list(TOOL_NAMES)
        self.matrices = {action: _identity(self.dimension) for action in self.actions}
        self.reward_vectors = {action: [0.0] * self.dimension for action in self.actions}

    def select_tool(self, query: str) -> str:
        features = extract_features(query)
        scores: dict[str, float] = {}
        for action in self.actions:
            matrix = self.matrices[action]
            theta = _solve(matrix, self.reward_vectors[action])
            inverse_times_features = _solve(matrix, features)
            uncertainty = math.sqrt(max(_dot(features, inverse_times_features), 0.0))
            scores[action] = _dot(theta, features) + self.alpha * uncertainty
        return max(self.actions, key=lambda action: (scores[action], -self.actions.index(action)))

    def update(self, query: str, action: str, reward: float) -> None:
        if action not in self.actions:
            raise ValueError(f"Unknown action: {action}")
        features = extract_features(query)
        matrix = self.matrices[action]
        for row in range(self.dimension):
            for column in range(self.dimension):
                matrix[row][column] += features[row] * features[column]
            self.reward_vectors[action][row] += reward * features[row]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "alpha": self.alpha,
                    "dimension": self.dimension,
                    "actions": self.actions,
                    "matrices": self.matrices,
                    "reward_vectors": self.reward_vectors,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> LinUCBPolicy:
        payload = json.loads(path.read_text(encoding="utf-8"))
        policy = cls(alpha=float(payload["alpha"]))
        if payload["dimension"] != policy.dimension or payload["actions"] != policy.actions:
            raise ValueError("Policy artifact does not match current feature/action schema.")
        policy.matrices = {
            action: [[float(value) for value in row] for row in matrix]
            for action, matrix in payload["matrices"].items()
        }
        policy.reward_vectors = {
            action: [float(value) for value in vector]
            for action, vector in payload["reward_vectors"].items()
        }
        return policy

