"""
MCTS proof tree. Each node is a sub-goal. UCB1 selection, backpropagation, pruning.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from pathlib import Path


class NodeStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    VERIFIED = "verified"
    PARTIAL = "partial"
    FAILED = "failed"
    PRUNED = "pruned"


@dataclass
class ProofAttempt:
    attempt_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str = ""
    is_formal: bool = False
    verifier_score: float = 0.0
    verifier_feedback: str = ""
    timestamp: str = ""
    iteration: int = 0


@dataclass
class ProofNode:
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: Optional[str] = None
    label: str = ""
    statement: str = ""
    branch: str = ""
    dependencies: list[str] = field(default_factory=list)
    status: NodeStatus = NodeStatus.OPEN
    attempts: list[ProofAttempt] = field(default_factory=list)
    best_score: float = 0.0
    visit_count: int = 0
    total_value: float = 0.0
    consecutive_failures: int = 0
    children: list[str] = field(default_factory=list)
    depth: int = 0

    @property
    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb1(self, parent_visits: int, C: float = 1.414) -> float:
        if self.status in (NodeStatus.VERIFIED, NodeStatus.PRUNED):
            return -float('inf')
        if self.visit_count == 0:
            return float('inf')
        return self.mean_value + C * math.sqrt(math.log(parent_visits) / self.visit_count)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "label": self.label,
            "statement": self.statement,
            "branch": self.branch,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "attempts": [
                {
                    "attempt_id": a.attempt_id,
                    "content": a.content[:500] + "..." if len(a.content) > 500 else a.content,
                    "is_formal": a.is_formal,
                    "verifier_score": a.verifier_score,
                    "verifier_feedback": a.verifier_feedback,
                    "iteration": a.iteration,
                }
                for a in self.attempts
            ],
            "best_score": self.best_score,
            "visit_count": self.visit_count,
            "total_value": self.total_value,
            "mean_value": round(self.mean_value, 4),
            "consecutive_failures": self.consecutive_failures,
            "children": self.children,
            "depth": self.depth,
        }


class ProofTree:
    def __init__(self, config):
        self.config = config
        self.nodes: dict[str, ProofNode] = {}
        self.root_id: Optional[str] = None
        self.iteration: int = 0
        self.history: list[dict] = []

    def create_root(self, label: str, statement: str) -> ProofNode:
        node = ProofNode(label=label, statement=statement, branch="ROOT", depth=0)
        self.nodes[node.node_id] = node
        self.root_id = node.node_id
        return node

    def add_child(
        self,
        parent_id: str,
        label: str,
        statement: str,
        branch: str = "",
        dependencies: list[str] | None = None,
    ) -> ProofNode:
        parent = self.nodes[parent_id]
        if parent.depth + 1 > self.config.mcts.max_depth:
            raise ValueError(f"Max depth {self.config.mcts.max_depth} exceeded")
        if len(parent.children) >= self.config.mcts.max_children:
            raise ValueError(f"Max children {self.config.mcts.max_children} exceeded for {parent_id}")

        node = ProofNode(
            parent_id=parent_id,
            label=label,
            statement=statement,
            branch=branch or f"{parent.branch}.{len(parent.children) + 1}",
            dependencies=dependencies or [],
            depth=parent.depth + 1,
        )
        self.nodes[node.node_id] = node
        parent.children.append(node.node_id)
        return node

    def select(self) -> Optional[ProofNode]:
        """UCB1 traversal from root to best leaf."""
        if self.root_id is None:
            return None

        current = self.nodes[self.root_id]

        while current.children:
            selectable = [
                self.nodes[cid] for cid in current.children
                if self.nodes[cid].status not in (NodeStatus.VERIFIED, NodeStatus.PRUNED)
            ]
            if not selectable:
                break

            parent_visits = max(current.visit_count, 1)
            best = max(selectable, key=lambda n: n.ucb1(parent_visits, self.config.mcts.exploration_constant))

            if best.visit_count == 0:
                return best

            current = best

        if current.status not in (NodeStatus.VERIFIED, NodeStatus.PRUNED):
            return current
        return None

    def record_attempt(self, node_id: str, attempt: ProofAttempt) -> None:
        node = self.nodes[node_id]
        node.attempts.append(attempt)
        node.visit_count += 1
        node.best_score = max(node.best_score, attempt.verifier_score)

        if attempt.verifier_score >= 1.0:
            node.status = NodeStatus.VERIFIED
            node.consecutive_failures = 0
        elif attempt.verifier_score >= 0.3:
            node.status = NodeStatus.PARTIAL
            node.consecutive_failures = 0
        else:
            node.consecutive_failures += 1
            if node.consecutive_failures >= self.config.mcts.max_failures_before_prune:
                node.status = NodeStatus.PRUNED
            else:
                node.status = NodeStatus.FAILED

        self._backpropagate(node_id, attempt.verifier_score)

        self.history.append({
            "iteration": self.iteration,
            "node_id": node_id,
            "node_label": node.label,
            "branch": node.branch,
            "score": attempt.verifier_score,
            "status": node.status.value,
        })

    def _backpropagate(self, node_id: str, value: float) -> None:
        """Propagate score up to root with geometric decay."""
        decay = self.config.mcts.backprop_decay
        current_value = value
        current_id = node_id

        while current_id is not None:
            node = self.nodes[current_id]
            node.total_value += current_value
            current_value *= decay
            current_id = node.parent_id

    def get_frontier(self) -> list[ProofNode]:
        frontier = []
        for node in self.nodes.values():
            if node.status in (NodeStatus.OPEN, NodeStatus.FAILED, NodeStatus.PARTIAL):
                if not node.children or all(
                    self.nodes[cid].status in (NodeStatus.VERIFIED, NodeStatus.PRUNED)
                    for cid in node.children
                ):
                    frontier.append(node)
        return frontier

    def is_complete(self) -> bool:
        if self.root_id is None:
            return False
        return self._check_complete(self.root_id)

    def _check_complete(self, node_id: str) -> bool:
        node = self.nodes[node_id]
        if node.status == NodeStatus.VERIFIED:
            return True
        if not node.children:
            return False
        return all(self._check_complete(cid) for cid in node.children)

    def progress_summary(self) -> dict:
        by_status = {}
        for node in self.nodes.values():
            by_status[node.status.value] = by_status.get(node.status.value, 0) + 1

        frontier = self.get_frontier()
        best = max(frontier, key=lambda n: n.best_score) if frontier else None

        return {
            "iteration": self.iteration,
            "total_nodes": len(self.nodes),
            "by_status": by_status,
            "frontier_size": len(frontier),
            "best_frontier": {
                "label": best.label if best else None,
                "score": best.best_score if best else 0,
                "branch": best.branch if best else None,
            },
            "tree_complete": self.is_complete(),
        }

    def to_json(self) -> str:
        return json.dumps({
            "root_id": self.root_id,
            "iteration": self.iteration,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "history": self.history[-50:],
            "summary": self.progress_summary(),
        }, indent=2)

    def save_snapshot(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def from_json(cls, data: str, config) -> "ProofTree":
        raw = json.loads(data)
        tree = cls(config)
        tree.root_id = raw["root_id"]
        tree.iteration = raw["iteration"]
        tree.history = raw.get("history", [])

        for nid, ndata in raw["nodes"].items():
            node = ProofNode(
                node_id=ndata["node_id"],
                parent_id=ndata["parent_id"],
                label=ndata["label"],
                statement=ndata["statement"],
                branch=ndata["branch"],
                dependencies=ndata["dependencies"],
                status=NodeStatus(ndata["status"]),
                best_score=ndata["best_score"],
                visit_count=ndata["visit_count"],
                total_value=ndata["total_value"],
                consecutive_failures=ndata["consecutive_failures"],
                children=ndata["children"],
                depth=ndata["depth"],
            )
            for adata in ndata.get("attempts", []):
                node.attempts.append(ProofAttempt(
                    attempt_id=adata["attempt_id"],
                    content=adata["content"],
                    is_formal=adata["is_formal"],
                    verifier_score=adata["verifier_score"],
                    verifier_feedback=adata["verifier_feedback"],
                    iteration=adata["iteration"],
                ))
            tree.nodes[nid] = node

        return tree
