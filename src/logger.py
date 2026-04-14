"""Structured experiment logging following the research plan spec."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class AgentLog:
    """Single agent turn within an episode."""
    agent_id: str
    agent_role: str
    model_size: str
    framework: str
    round: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    output_raw: str = ""
    output_parsed: dict[str, Any] = field(default_factory=dict)
    schema_valid: bool = True
    solution: str = ""
    confidence: float = 0.0


@dataclass
class EpisodeLog:
    """Complete record of one deliberation episode."""
    experiment_id: str
    episode_id: str
    task_id: str
    benchmark: str
    difficulty: str
    gamma: float
    n_agents: int
    tier: int = 1
    # Framework assignment
    frameworks_available: list[str] = field(default_factory=list)
    framework_assignments: list[str] = field(default_factory=list)
    # Diversity metrics
    diversity_score: float = 0.0
    framework_coverage: float = 0.0
    shannon_entropy: float = 0.0
    # Population distribution
    population_distribution: dict[str, float] = field(default_factory=dict)
    # Agent-level logs
    agent_logs: list[AgentLog] = field(default_factory=list)
    # Outcome
    ground_truth: str = ""
    solutions: list[str] = field(default_factory=list)
    correct_count: int = 0
    # Cost
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    wall_time_s: float = 0.0
    # Escalation
    escalation_triggered: bool = False
    escalation_condition: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["agent_logs"] = [asdict(a) for a in self.agent_logs]
        return d


class ExperimentLogger:
    """Accumulates episode logs and writes to JSON."""

    def __init__(self, experiment_id: str, output_dir: Path):
        self.experiment_id = experiment_id
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episodes: list[EpisodeLog] = []
        self._start_time = time.time()

    def add_episode(self, episode: EpisodeLog) -> None:
        self.episodes.append(episode)

    def save(self, filename: str | None = None) -> Path:
        if filename is None:
            filename = f"{self.experiment_id}.json"
        path = self.output_dir / filename
        data = {
            "experiment_id": self.experiment_id,
            "n_episodes": len(self.episodes),
            "total_time_s": time.time() - self._start_time,
            "total_cost_usd": sum(e.total_cost_usd for e in self.episodes),
            "episodes": [e.to_dict() for e in self.episodes],
        }
        path.write_text(json.dumps(data, indent=2, default=str))
        return path

    def summary(self) -> dict[str, Any]:
        if not self.episodes:
            return {}
        gammas = sorted(set(e.gamma for e in self.episodes))
        summary: dict[str, Any] = {"gamma_levels": gammas, "per_gamma": {}}
        for g in gammas:
            eps = [e for e in self.episodes if e.gamma == g]
            d_scores = [e.diversity_score for e in eps]
            summary["per_gamma"][g] = {
                "n_episodes": len(eps),
                "mean_diversity": sum(d_scores) / len(d_scores) if d_scores else 0,
                "mean_cost": sum(e.total_cost_usd for e in eps) / len(eps) if eps else 0,
            }
        return summary
