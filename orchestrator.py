#!/usr/bin/env python3
"""
Main agent loop: select → expand → attempt → verify → update → repeat.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from config import Config
from proof_tree import ProofTree, ProofNode, ProofAttempt, NodeStatus
from llm_interface import LLMInterface
from verifier import Verifier


class Orchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.problem_context = self._load_problem()
        self.llm = LLMInterface(config.llm)
        self.tree = ProofTree(config)
        self.verifier = Verifier(config.verifier, self.llm, self.problem_context)
        self._init_tree_from_problem()

    def _load_problem(self) -> str:
        path = self.config.orchestrator.problem_path
        if not path.exists():
            raise FileNotFoundError(f"Problem file not found: {path}")
        return path.read_text()

    def _init_tree_from_problem(self) -> None:
        root = self.tree.create_root(
            label="Navier-Stokes Existence & Smoothness",
            statement=(
                "For smooth, divergence-free initial data u_0 on R^3 with sufficient "
                "decay, there exists a unique smooth solution u ∈ C^∞(R^3 × [0,∞)) "
                "to the incompressible Navier-Stokes equations."
            ),
        )

        # Branch A: Local Existence (known — foundation)
        a = self.tree.add_child(root.node_id, "Local Existence",
            "Local-in-time existence and uniqueness of mild solutions via Fujita-Kato, "
            "with regularity on the existence interval and continuation criterion.",
            branch="A")
        self.tree.add_child(a.node_id, "Fujita-Kato mild solution",
            "For u_0 ∈ L^3(R^3) divergence-free, there exists T > 0 and a unique "
            "mild solution u ∈ C([0,T]; L^3) to NS.",
            branch="A.1")
        self.tree.add_child(a.node_id, "Regularity on existence interval",
            "The mild solution from A.1 is in C^∞(R^3 × (0,T)).",
            branch="A.2")
        self.tree.add_child(a.node_id, "Continuation criterion",
            "The solution extends past T if and only if lim sup_{t→T} ||u(t)||_{L^3} < ∞.",
            branch="A.3")

        # Branch B: Energy Estimates
        b = self.tree.add_child(root.node_id, "A Priori Energy Estimates",
            "Establish energy inequalities and higher-order estimates.",
            branch="B")
        self.tree.add_child(b.node_id, "Energy inequality",
            "||u(t)||_{L^2}^2 + 2ν ∫_0^t ||∇u||_{L^2}^2 ds ≤ ||u_0||_{L^2}^2.",
            branch="B.1")
        self.tree.add_child(b.node_id, "Higher-order bootstrap",
            "If ||u||_{L^∞([0,T]; L^3)} < ∞, then u ∈ H^k for all k.",
            branch="B.2")

        # Branch C: Blow-Up Exclusion (the hard part)
        c = self.tree.add_child(root.node_id, "Blow-Up Exclusion",
            "Prove that no finite-time blow-up occurs for smooth initial data.",
            branch="C")

        self.tree.add_child(c.node_id, "Scaling analysis",
            "Analyze self-similar blow-up profiles and show they cannot exist.",
            branch="C.1")
        c2 = self.tree.add_child(c.node_id, "Critical norm regularity",
            "Show blow-up requires critical norms to diverge, then show they can't.",
            branch="C.2")
        self.tree.add_child(c.node_id, "Vorticity geometric constraints",
            "Use vorticity structure to bound blow-up rate.",
            branch="C.3")
        self.tree.add_child(c.node_id, "Frequency-space approach",
            "Use Littlewood-Paley decomposition to control energy cascade.",
            branch="C.4")

        # C.2 sub-branches (most explored direction in literature)
        self.tree.add_child(c2.node_id, "ESS backward uniqueness",
            "Escauriaza-Seregin-Šverák: if u ∈ L^∞([0,T); L^{3,∞}), then u is regular at T.",
            branch="C.2.1")
        self.tree.add_child(c2.node_id, "Strengthen to L^3",
            "Show that blow-up at T implies ||u(t)||_{L^3} → ∞ as t → T.",
            branch="C.2.2")
        self.tree.add_child(c2.node_id, "Contradiction with energy",
            "Show L^3 blow-up contradicts energy estimates (the gap).",
            branch="C.2.3")

        self._log("Tree initialized with Navier-Stokes decomposition")
        self._log(f"  Nodes: {len(self.tree.nodes)}, Frontier: {len(self.tree.get_frontier())}")

    def run(self) -> None:
        budget = self.config.orchestrator.budget
        self._log(f"\n{'='*60}")
        self._log(f"  autoproof — starting proof exploration")
        self._log(f"  Budget: {budget} iterations")
        self._log(f"  Lean 4: {'available' if self.verifier._lean_available else 'not available (using LLM judge)'}")
        self._log(f"  LLM: {self.config.llm.model}")
        self._log(f"{'='*60}\n")

        for i in range(budget):
            self.tree.iteration = i + 1
            start = time.time()

            # 1. SELECT
            node = self.tree.select()
            if node is None:
                self._log(f"\n[iter {i+1}] No selectable nodes. All branches exhausted or verified.")
                break

            self._log(f"\n[iter {i+1}] SELECTED: [{node.branch}] {node.label}")
            self._log(f"  UCB stats: visits={node.visit_count}, mean={node.mean_value:.3f}, "
                      f"failures={node.consecutive_failures}")

            # 2. EXPAND (if node has been visited but has no children yet)
            if node.visit_count > 0 and not node.children and node.status == NodeStatus.PARTIAL:
                self._log(f"  → Decomposing into sub-goals...")
                self._expand_node(node)
                new_node = self.tree.select()
                if new_node and new_node.node_id != node.node_id:
                    node = new_node
                    self._log(f"  → Now targeting: [{node.branch}] {node.label}")

            # 3. ATTEMPT
            self._log(f"  → Generating proof attempt...")
            node.status = NodeStatus.IN_PROGRESS

            parent_context = ""
            if node.parent_id:
                parent = self.tree.nodes[node.parent_id]
                parent_context = f"Parent goal [{parent.branch}]: {parent.statement}"

            previous_attempts = [
                {
                    "content": a.content[:300],
                    "score": a.verifier_score,
                    "feedback": a.verifier_feedback,
                }
                for a in node.attempts[-3:]
            ]

            proof_content = self.llm.generate_proof(
                problem_context=self.problem_context,
                node_label=node.label,
                node_statement=node.statement,
                parent_context=parent_context,
                previous_attempts=previous_attempts,
                tree_summary=self.tree.progress_summary(),
            )

            # 4. VERIFY
            self._log(f"  → Verifying...")
            result = self.verifier.verify(
                node_statement=node.statement,
                proof_content=proof_content,
            )

            score = result.score
            if result.is_novel:
                score = min(1.0, score + self.config.mcts.bonus_on_novelty)
                self._log(f"  → NOVEL approach detected! Bonus applied.")

            # 5. UPDATE
            attempt = ProofAttempt(
                content=proof_content,
                is_formal=result.is_formal,
                verifier_score=score,
                verifier_feedback=result.feedback,
                timestamp=datetime.now(timezone.utc).isoformat(),
                iteration=i + 1,
            )
            self.tree.record_attempt(node.node_id, attempt)

            elapsed = time.time() - start
            status_icon = {
                NodeStatus.VERIFIED: "✓",
                NodeStatus.PARTIAL: "◐",
                NodeStatus.FAILED: "✗",
                NodeStatus.PRUNED: "⊘",
            }.get(node.status, "?")

            self._log(f"  {status_icon} Score: {score:.2f} | Status: {node.status.value} | {elapsed:.1f}s")
            self._log(f"  Feedback: {result.feedback[:150]}")

            # 6. SNAPSHOT
            if (i + 1) % self.config.orchestrator.snapshot_every == 0:
                snap_path = self.config.orchestrator.log_dir / f"tree_iter_{i+1:04d}.json"
                self.tree.save_snapshot(snap_path)
                self._log(f"  → Snapshot saved: {snap_path}")

            if self.tree.is_complete():
                self._log(f"\n{'='*60}")
                self._log(f"  PROOF COMPLETE at iteration {i+1}!")
                self._log(f"{'='*60}")
                break

            summary = self.tree.progress_summary()
            self._log(f"  Progress: {json.dumps(summary['by_status'])}")

        self._final_report()

    def _expand_node(self, node: ProofNode) -> None:
        existing = [self.tree.nodes[cid].label for cid in node.children]
        proposals = self.llm.propose_decomposition(
            node_label=node.label,
            node_statement=node.statement,
            problem_context=self.problem_context,
            existing_children=existing,
        )
        for prop in proposals[:self.config.mcts.max_children - len(node.children)]:
            try:
                child = self.tree.add_child(
                    parent_id=node.node_id,
                    label=prop.get("label", "unnamed"),
                    statement=prop.get("statement", ""),
                )
                self._log(f"    + Added sub-goal [{child.branch}]: {child.label}")
            except ValueError as e:
                self._log(f"    ! Could not add sub-goal: {e}")

    def _final_report(self) -> None:
        summary = self.tree.progress_summary()

        self._log(f"\n{'='*60}")
        self._log(f"  autoproof — FINAL REPORT")
        self._log(f"{'='*60}")
        self._log(f"  Iterations: {self.tree.iteration}")
        self._log(f"  Nodes: {summary['total_nodes']}")
        self._log(f"  Status: {json.dumps(summary['by_status'], indent=2)}")
        self._log(f"  Complete: {summary['tree_complete']}")

        if summary['best_frontier']['label']:
            self._log(f"\n  Most promising frontier node:")
            self._log(f"    [{summary['best_frontier']['branch']}] {summary['best_frontier']['label']}")
            self._log(f"    Best score: {summary['best_frontier']['score']:.3f}")

        verified = [n for n in self.tree.nodes.values() if n.status == NodeStatus.VERIFIED]
        if verified:
            self._log(f"\n  Verified sub-goals ({len(verified)}):")
            for v in verified:
                self._log(f"    ✓ [{v.branch}] {v.label}")

        pruned = [n for n in self.tree.nodes.values() if n.status == NodeStatus.PRUNED]
        if pruned:
            self._log(f"\n  Pruned branches ({len(pruned)}):")
            for p in pruned:
                self._log(f"    ⊘ [{p.branch}] {p.label} (after {len(p.attempts)} attempts)")

        final_path = self.config.orchestrator.log_dir / "tree_final.json"
        self.tree.save_snapshot(final_path)
        self._log(f"\n  Final tree saved: {final_path}")

        history_path = self.config.orchestrator.log_dir / "history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(self.tree.history, indent=2))
        self._log(f"  History saved: {history_path}")

    def _log(self, msg: str) -> None:
        if self.config.orchestrator.verbose:
            print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser(description="autoproof — automated theorem proving via MCTS")
    parser.add_argument("--problem", type=str, default="problem.md")
    parser.add_argument("--budget", type=int, default=None, help="overrides config")
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument("--resume", type=str, default=None, help="resume from snapshot JSON")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = Config.from_toml(args.config)
    config.orchestrator.problem_path = Path(args.problem)
    if args.budget is not None:
        config.orchestrator.budget = args.budget
    config.orchestrator.verbose = not args.quiet

    orch = Orchestrator(config)

    if args.resume:
        snapshot = Path(args.resume).read_text()
        orch.tree = ProofTree.from_json(snapshot, config)
        print(f"Resumed from {args.resume} at iteration {orch.tree.iteration}")

    orch.run()


if __name__ == "__main__":
    main()
