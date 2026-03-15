"""
MLX local inference provider for fully offline proof exploration.

Supports two modes:
  1. Single model:  One model handles both proof generation and judging
  2. Dual model:    Specialized prover (DeepSeek-Prover) + general judge (Qwen/Llama)

Requires: pip install mlx-lm (macOS Apple Silicon only)

Recommended models:
  Prover:  mlx-community/DeepSeek-Prover-V2-7B-4bit     (~4GB, Lean 4 specialized)
  Judge:   mlx-community/Qwen2.5-7B-Instruct-4bit       (~4GB, general reasoning)
  Both:    mlx-community/Qwen2.5-14B-Instruct-4bit       (~8GB, good at both)

On an M4 MacBook Pro with 24GB+ unified memory, you can comfortably run
the dual-model setup with both loaded simultaneously.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MLXModelConfig:
    """Configuration for a single MLX model."""
    model_id: str = "mlx-community/DeepSeek-Prover-V2-7B-4bit"
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.05
    max_kv_size: Optional[int] = None  # None = unlimited, set to e.g. 2048 for RAM savings


@dataclass
class MLXConfig:
    """Configuration for the MLX provider."""
    # Primary prover model (generates proof attempts)
    prover: MLXModelConfig = field(default_factory=lambda: MLXModelConfig(
        model_id="mlx-community/DeepSeek-Prover-V2-7B-4bit",
        temperature=0.7,
    ))
    # Judge model (evaluates proofs). If None, uses prover for both.
    judge: Optional[MLXModelConfig] = field(default_factory=lambda: MLXModelConfig(
        model_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
        temperature=0.2,
    ))
    # If True, keep both models in memory (needs ~8-10GB for dual 7B-4bit)
    # If False, unload prover before loading judge and vice versa (slower but less RAM)
    keep_both_loaded: bool = True
    # Verbose: print generation stats
    verbose: bool = True


class MLXProvider:
    """
    Local MLX inference provider.

    Usage:
        provider = MLXProvider(MLXConfig())
        text = provider.generate("Prove that...", role="prover")
        judgment = provider.generate("Evaluate this proof...", role="judge")
    """

    def __init__(self, config: MLXConfig):
        self.config = config
        self._models: dict[str, tuple] = {}  # role -> (model, tokenizer)
        self._mlx_lm = None

    def _ensure_mlx_lm(self):
        """Lazy import mlx_lm."""
        if self._mlx_lm is not None:
            return
        try:
            import mlx_lm
            self._mlx_lm = mlx_lm
        except ImportError:
            raise RuntimeError(
                "mlx-lm not installed. Run: pip install mlx-lm\n"
                "Note: mlx-lm requires macOS on Apple Silicon (M1+)."
            )

    def _get_model(self, role: str) -> tuple:
        """
        Load or retrieve a cached model for the given role.

        Roles: "prover" or "judge"
        """
        if role in self._models:
            return self._models[role]

        self._ensure_mlx_lm()

        if role == "judge" and self.config.judge is not None:
            model_config = self.config.judge
        else:
            model_config = self.config.prover

        # If not keeping both loaded and switching roles, free the other
        if not self.config.keep_both_loaded:
            other_role = "judge" if role == "prover" else "prover"
            if other_role in self._models:
                if self.config.verbose:
                    print(f"  [MLX] Unloading {other_role} model to free memory...")
                del self._models[other_role]
                import gc
                gc.collect()

        if self.config.verbose:
            print(f"  [MLX] Loading {role} model: {model_config.model_id}")
            t0 = time.time()

        model, tokenizer = self._mlx_lm.load(model_config.model_id)

        if self.config.verbose:
            print(f"  [MLX] Loaded in {time.time() - t0:.1f}s")

        self._models[role] = (model, tokenizer)
        return (model, tokenizer)

    def generate(
        self,
        system: str,
        user: str,
        role: str = "prover",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text using the specified model role.

        Args:
            system: System prompt (role/instructions)
            user: User message (the actual query)
            role: "prover" for proof generation, "judge" for evaluation
            temperature: Override model temperature
            max_tokens: Override max tokens

        Returns:
            Generated text content
        """
        self._ensure_mlx_lm()
        model, tokenizer = self._get_model(role)

        model_config = (
            self.config.judge if role == "judge" and self.config.judge else self.config.prover
        )
        temp = temperature if temperature is not None else model_config.temperature
        max_tok = max_tokens if max_tokens is not None else model_config.max_tokens

        # Build the prompt via chat template
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        # Apply chat template if available
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback: simple concatenation for base models without chat template
            prompt = f"### System:\n{system}\n\n### User:\n{user}\n\n### Assistant:\n"

        if self.config.verbose:
            print(f"  [MLX] Generating ({role}, temp={temp}, max_tokens={max_tok})...")
            t0 = time.time()

        # Build generation kwargs
        gen_kwargs = {
            "temp": temp,
            "max_tokens": max_tok,
            "top_p": model_config.top_p,
            "repetition_penalty": model_config.repetition_penalty,
        }
        if model_config.max_kv_size is not None:
            gen_kwargs["max_kv_size"] = model_config.max_kv_size

        response = self._mlx_lm.generate(
            model, tokenizer,
            prompt=prompt,
            verbose=False,
            **gen_kwargs,
        )

        if self.config.verbose:
            elapsed = time.time() - t0
            tokens = len(tokenizer.encode(response)) if response else 0
            tps = tokens / elapsed if elapsed > 0 else 0
            print(f"  [MLX] Generated {tokens} tokens in {elapsed:.1f}s ({tps:.1f} tok/s)")

        return response

    def unload_all(self):
        """Free all loaded models from memory."""
        self._models.clear()
        import gc
        gc.collect()
        if self.config.verbose:
            print("  [MLX] All models unloaded")


class MLXServerProvider:
    """
    Alternative: connect to a local mlx_lm.server running OpenAI-compatible API.

    Start the server with:
        mlx_lm.server --model mlx-community/DeepSeek-Prover-V2-7B-4bit --port 8081

    This is useful if you want to:
    - Run the model in a separate process
    - Use the same model from multiple scripts
    - Avoid loading/unloading overhead
    """

    def __init__(self, base_url: str = "http://localhost:8081", model: str = "default"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(
        self,
        system: str,
        user: str,
        role: str = "prover",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        import urllib.request
        import urllib.error

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature or 0.7,
            "max_tokens": max_tokens or 4096,
        }

        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"]
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Could not connect to MLX server at {self.base_url}. "
                f"Start it with: mlx_lm.server --model <model> --port 8081\n"
                f"Error: {e}"
            )
