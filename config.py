"""Configuration dataclasses. Loaded from TOML, with defaults."""

from dataclasses import dataclass, field
from pathlib import Path
import tomllib


@dataclass
class MCTSConfig:
    exploration_constant: float = 1.414
    max_depth: int = 12
    max_children: int = 5
    max_failures_before_prune: int = 3
    penalty_on_failure: float = -0.3
    bonus_on_novelty: float = 0.2
    backprop_decay: float = 0.9


@dataclass
class VerifierConfig:
    use_lean4: bool = False
    lean4_project_path: Path = Path("lean_project")
    lean4_timeout_seconds: int = 120
    llm_judge_model: str = "claude-sonnet-4-20250514"
    llm_judge_temperature: float = 0.2
    require_formal_for_complete: bool = True


@dataclass
class LLMConfig:
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    system_prompt_path: Path = Path("problem.md")


@dataclass
class OrchestratorConfig:
    budget: int = 100
    snapshot_every: int = 5
    log_dir: Path = Path("logs")
    problem_path: Path = Path("problem.md")
    verbose: bool = True

    def __post_init__(self):
        self.log_dir = Path(self.log_dir)
        self.problem_path = Path(self.problem_path)


@dataclass
class Config:
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)

    @classmethod
    def from_toml(cls, path: str = "config.toml") -> "Config":
        config = cls()
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
            if "mcts" in data:
                for k, v in data["mcts"].items():
                    if hasattr(config.mcts, k):
                        setattr(config.mcts, k, v)
            if "verifier" in data:
                for k, v in data["verifier"].items():
                    if hasattr(config.verifier, k):
                        setattr(config.verifier, k, v)
            if "llm" in data:
                for k, v in data["llm"].items():
                    if hasattr(config.llm, k):
                        setattr(config.llm, k, v)
            if "orchestrator" in data:
                for k, v in data["orchestrator"].items():
                    if hasattr(config.orchestrator, k):
                        if k in ("log_dir", "problem_path"):
                            v = Path(v)
                        setattr(config.orchestrator, k, v)
        except FileNotFoundError:
            pass
        return config
