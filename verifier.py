"""
Dual-mode verifier: Lean 4 (formal, binary) or LLM-judge (informal, scored 0–1).
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from config import VerifierConfig
from llm_interface import LLMInterface


@dataclass
class VerificationResult:
    score: float
    feedback: str
    is_formal: bool
    is_novel: bool = False
    lean_errors: str = ""


class Verifier:
    def __init__(self, config: VerifierConfig, llm: LLMInterface, problem_context: str):
        self.config = config
        self.llm = llm
        self.problem_context = problem_context
        self._lean_available = self._check_lean()

    def _check_lean(self) -> bool:
        if not self.config.use_lean4:
            return False
        try:
            result = subprocess.run(["lean", "--version"], capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def verify(self, node_statement: str, proof_content: str, force_formal: bool = False) -> VerificationResult:
        is_lean_proof = self._detect_lean_syntax(proof_content)

        if is_lean_proof and self._lean_available:
            return self._verify_lean(node_statement, proof_content)
        elif force_formal and self._lean_available:
            result = self._verify_llm(node_statement, proof_content)
            result.feedback = "[formal verification requested but proof is NL] " + result.feedback
            return result
        else:
            return self._verify_llm(node_statement, proof_content)

    def _detect_lean_syntax(self, content: str) -> bool:
        lean_markers = ["theorem ", "lemma ", "def ", ":= by", "sorry", "import Mathlib"]
        return any(marker in content for marker in lean_markers)

    def _verify_lean(self, statement: str, proof: str) -> VerificationResult:
        project_path = self.config.lean4_project_path

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.lean',
            dir=project_path / "AutoProof",
            delete=False
        ) as f:
            f.write(f"import Mathlib\nimport AutoProof.Defs\n\n-- Goal: {statement}\n\n{proof}\n")
            temp_path = f.name

        try:
            result = subprocess.run(
                ["lake", "env", "lean", temp_path],
                capture_output=True, text=True,
                timeout=self.config.lean4_timeout_seconds,
                cwd=str(project_path),
            )

            if result.returncode == 0:
                return VerificationResult(score=1.0, feedback="Lean 4 verification PASSED.", is_formal=True)
            else:
                errors = result.stderr or result.stdout
                if len(errors) > 1000:
                    errors = errors[:500] + "\n...\n" + errors[-500:]
                return VerificationResult(score=0.0, feedback="Lean 4 verification FAILED.", is_formal=True, lean_errors=errors)
        except subprocess.TimeoutExpired:
            return VerificationResult(
                score=0.0,
                feedback=f"Lean 4 timed out after {self.config.lean4_timeout_seconds}s",
                is_formal=True, lean_errors="timeout",
            )
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def _verify_llm(self, statement: str, proof: str) -> VerificationResult:
        score, feedback = self.llm.judge_proof(
            node_statement=statement,
            proof_content=proof,
            problem_context=self.problem_context,
        )
        return VerificationResult(
            score=score,
            feedback=feedback,
            is_formal=False,
            is_novel=feedback.startswith("[NOVEL]"),
        )
