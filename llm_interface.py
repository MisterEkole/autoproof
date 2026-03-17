"""
LLM provider abstraction. Two roles: PROVER (generates proofs) and JUDGE (scores them).
Supports Anthropic, OpenAI, local MLX, and MLX server.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from config import LLMConfig


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict


class LLMInterface:
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY", "")

        if self.config.provider == "anthropic":
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise RuntimeError("pip install anthropic")
        elif self.config.provider == "openai":
            try:
                import openai
                self._client = openai.OpenAI(
                    api_key=api_key or os.environ.get("OPENAI_API_KEY", "")
                )
            except ImportError:
                raise RuntimeError("pip install openai")
        elif self.config.provider == "local_mlx":
            from mlx_provider import MLXProvider, MLXConfig, MLXModelConfig
            mlx_cfg = MLXConfig(
                prover=MLXModelConfig(
                    model_id=self.config.model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                ),
                judge=MLXModelConfig(
                    model_id=os.environ.get("AUTOPROOF_JUDGE_MODEL", self.config.model),
                    temperature=0.2,
                    max_tokens=self.config.max_tokens,
                ),
                keep_both_loaded=os.environ.get("AUTOPROOF_DUAL_LOADED", "1") == "1",
            )
            self._client = MLXProvider(mlx_cfg)
        elif self.config.provider == "mlx_server":
            from mlx_provider import MLXServerProvider
            base_url = os.environ.get("MLX_SERVER_URL", "http://localhost:8081")
            self._client = MLXServerProvider(base_url=base_url, model=self.config.model)
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

        return self._client

    def _call(self, system: str, user: str, temperature: float | None = None, role: str = "prover") -> LLMResponse:
        client = self._get_client()
        temp = temperature if temperature is not None else self.config.temperature

        if self.config.provider in ("local_mlx", "mlx_server"):
            content = client.generate(
                system=system, user=user, role=role,
                temperature=temp, max_tokens=self.config.max_tokens,
            )
            return LLMResponse(content=content, model=self.config.model, usage={})
        elif self.config.provider == "anthropic":
            resp = client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=temp,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return LLMResponse(
                content=resp.content[0].text,
                model=resp.model,
                usage={"input": resp.usage.input_tokens, "output": resp.usage.output_tokens},
            )
        elif self.config.provider == "openai":
            resp = client.chat.completions.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=temp,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return LLMResponse(
                content=resp.choices[0].message.content,
                model=resp.model,
                usage={"input": resp.usage.prompt_tokens, "output": resp.usage.completion_tokens},
            )

    # ── PROVER ───────────────────────────────────────────────────

    def generate_proof(
        self,
        problem_context: str,
        node_label: str,
        node_statement: str,
        parent_context: str,
        previous_attempts: list[dict],
        tree_summary: dict,
    ) -> str:
        system = f"""You are a research mathematician working on automated theorem proving.
You are attempting to prove sub-goals of a larger theorem.

PROBLEM CONTEXT:
{problem_context}

PROOF TREE STATE:
- Total nodes: {tree_summary.get('total_nodes', 0)}
- Verified: {tree_summary.get('by_status', {}).get('verified', 0)}
- Open: {tree_summary.get('by_status', {}).get('open', 0)}
- Failed: {tree_summary.get('by_status', {}).get('failed', 0)}
- Pruned: {tree_summary.get('by_status', {}).get('pruned', 0)}

RULES:
1. Be mathematically rigorous. Every step must be justified.
2. If you need an unproven lemma, clearly mark it as [UNPROVEN LEMMA: ...]
3. If a step follows from a known result, cite it: [KNOWN: Fujita-Kato 1964]
4. Structure your proof with clear numbered steps.
5. If you believe this sub-goal is unprovable with current techniques, say so explicitly.
6. If previous attempts failed, try a DIFFERENT approach.

OUTPUT FORMAT:
<proof_attempt>
<approach>Brief description of your strategy</approach>
<steps>
Step 1: ...
Step 2: ...
...
</steps>
<confidence>0.0-1.0 self-assessment</confidence>
<unproven_lemmas>List any sub-goals this creates</unproven_lemmas>
<alternative_if_fails>What to try next if this doesn't work</alternative_if_fails>
</proof_attempt>"""

        previous_str = ""
        if previous_attempts:
            previous_str = "\n\nPREVIOUS FAILED ATTEMPTS:\n"
            for i, att in enumerate(previous_attempts[-3:], 1):
                previous_str += f"\n--- Attempt {i} (score: {att.get('score', 0)}) ---\n"
                previous_str += f"Feedback: {att.get('feedback', 'none')}\n"
                previous_str += f"Content: {att.get('content', '')[:300]}...\n"

        user = f"""CURRENT SUB-GOAL:
Label: {node_label}
Statement: {node_statement}

PARENT CONTEXT:
{parent_context}
{previous_str}

Generate a rigorous proof attempt for this sub-goal."""

        return self._call(system, user).content

    # ── JUDGE ────────────────────────────────────────────────────

    def judge_proof(
        self,
        node_statement: str,
        proof_content: str,
        problem_context: str,
    ) -> tuple[float, str]:
        """Score a proof attempt. Returns (score 0–1, feedback)."""
        system = """You are a rigorous mathematics proof verifier. Evaluate proof attempts
for logical correctness, completeness, and rigor.

Be STRICT. Catch: circular reasoning, unjustified steps, misapplied theorems,
wrong bounds, swapped quantifiers, gaps papered over with "clearly" or "it follows".

SCORING:
- 1.0: Complete, every step justified, no gaps.
- 0.7: Correct structure, minor fillable gaps.
- 0.5: Viable direction, significant gaps.
- 0.3: Some correct ideas, fundamental issues.
- 0.0: Logically invalid or completely wrong.

Respond in this exact JSON format:
{"score": <float>, "feedback": "<detailed explanation>", "gaps": ["<gap1>", ...], "is_novel": <bool>}"""

        user = f"""STATEMENT TO PROVE:
{node_statement}

PROOF ATTEMPT:
{proof_content}

PROBLEM CONTEXT (for reference):
{problem_context[:2000]}

Evaluate this proof attempt. Be strict."""

        resp = self._call(system, user, temperature=0.2, role="judge")
        content = resp.content

        try:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            result = json.loads(json_match.group() if json_match else content.strip())
            score = float(result.get("score", 0.0))
            feedback = result.get("feedback", "No feedback")
            if result.get("is_novel", False):
                feedback = "[NOVEL] " + feedback
            return (min(max(score, 0.0), 1.0), feedback)
        except (json.JSONDecodeError, ValueError, AttributeError):
            score_match = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', content)
            feedback_match = re.search(r'"feedback"\s*:\s*"([^"]{0,300})', content)
            if score_match:
                score = min(max(float(score_match.group(1)), 0.0), 1.0)
                feedback = feedback_match.group(1) if feedback_match else "Partial parse"
                return (score, feedback)
            # Plain-text fallback: infer score from keywords
            low = re.search(
                r'\b(invalid|incorrect|wrong|circular|fails?|incomplete|error|gap|missing|flawed)\b',
                content, re.IGNORECASE,
            )
            high = re.search(
                r'\b(correct|valid|complete|sound|rigorous|proven|verified)\b',
                content, re.IGNORECASE,
            )
            inferred_score = 0.3 if (high and not low) else 0.1
            feedback = content.strip()[:300] if content.strip() else "No feedback"
            return (inferred_score, feedback)

    # ── DECOMPOSER ───────────────────────────────────────────────

    def propose_decomposition(
        self,
        node_label: str,
        node_statement: str,
        problem_context: str,
        existing_children: list[str],
    ) -> list[dict]:
        """Ask the LLM to break a node into 2–4 sub-goals."""
        system = """You are a research mathematician decomposing proof goals into sub-goals.

Break the given statement into 2-4 independent or sequential sub-goals that together
establish the main result. Each should be precisely stated and simpler than the parent.
Mark known results as [KNOWN].

Respond ONLY with a JSON array:
[{"label": "short name", "statement": "precise mathematical statement", "known": false}, ...]"""

        existing_str = ""
        if existing_children:
            existing_str = f"\n\nEXISTING SUB-GOALS (don't duplicate):\n" + "\n".join(f"- {c}" for c in existing_children)

        user = f"""GOAL TO DECOMPOSE:
Label: {node_label}
Statement: {node_statement}

PROBLEM CONTEXT:
{problem_context[:3000]}
{existing_str}

Propose sub-goals."""

        resp = self._call(system, user, temperature=0.5, role="judge")

        try:
            json_match = re.search(r'\[.*\]', resp.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(resp.content)
        except json.JSONDecodeError:
            return []
