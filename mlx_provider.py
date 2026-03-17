"""MLX local inference provider"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional
import warnings

# Patterns in a model ID that signal it is a base/completion model (not instruct).
# Checked case-insensitively.  If ANY pattern matches, the model is treated as
# a base model and gets a plain-text prompt + BOS/EOS post-processing.
_BASE_MODEL_PATTERNS = [
    r"prover",          # DeepSeek-Prover, etc.
    r"(?<![_-])base",   # ...-Base, not "database"
    r"-v\d+(\.\d+)?$",  # bare versioned releases with no role suffix
]

# Special tokens emitted by DeepSeek (and similar) base models when they
# "continue" the conversation by generating the next turn.
_BOS_MARKER = "<｜begin▁of▁sentence｜>"
_COMPLETION_STOP_MARKERS = [
    _BOS_MARKER,
    "<｜end▁of▁sentence｜>",
    "<｜User｜>",
    "<｜Assistant｜>",
    "<｜System｜>",
]


def _is_base_model(model_id: str, tokenizer) -> bool:
    """Return True if the model should be treated as a base/completion model.

    Priority order:
    1. Explicit instruct/chat signals in the model ID → always instruct.
    2. Known base-model patterns in the model ID → always base.
    3. Tokenizer has no usable chat_template → treat as base.
    """
    name = model_id.lower()

    # Explicit instruct/chat signals take priority.
    # Use word-boundary-aware checks to avoid false matches like "4bit" -> "it".
    instruct_signals = ("instruct", "chat", "-sft")
    if any(kw in name for kw in instruct_signals):
        return False
    # "-it" suffix (e.g. mistral-7b-it) but not "4bit"
    if re.search(r"-it\b", name):
        return False

    # Known base-model name patterns
    for pat in _BASE_MODEL_PATTERNS:
        if re.search(pat, name, re.IGNORECASE):
            return True

    # Fall back to tokenizer capability
    return not (hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template)


def _build_prompt(system: str, user: str, tokenizer, is_base: bool) -> str:
    """Format the prompt appropriately for the model type."""
    if not is_base:
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
    # Plain completion format for base models — no special tokens, just clear
    # section headers so the model knows where to write its answer.
    return f"System: {system}\n\nProblem:\n{user}\n\nSolution:\n"


def _postprocess(text: str, is_base: bool) -> str:
    """Clean up the generated text.

    For base models, truncate at the first sign of a new conversation turn
    (the model generating what comes *after* its answer) and strip any
    remaining special tokens.
    For instruct models the output is already clean; just strip whitespace.
    """
    if not is_base:
        return text.strip()

    # Truncate at the earliest stop marker
    cut = len(text)
    for marker in _COMPLETION_STOP_MARKERS:
        idx = text.find(marker)
        if idx != -1 and idx < cut:
            cut = idx
    text = text[:cut]

    # Remove any stray special-token strings that survived
    for marker in _COMPLETION_STOP_MARKERS:
        text = text.replace(marker, "")

    return text.strip()


@dataclass
class MLXModelConfig:
    """Configuration for a single MLX model."""
    model_id: str = "mlx-community/Qwen2.5-Math-7B-Instruct-4bit"
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.05
    max_kv_size: Optional[int] = None  # None = unlimited


@dataclass
class MLXConfig:
    """Configuration for the MLX provider."""
    # Proof generation model
    prover: MLXModelConfig = field(default_factory=lambda: MLXModelConfig(
        model_id="mlx-community/Qwen2.5-Math-7B-Instruct-4bit",
        temperature=0.7,
    ))
    # Evaluation model; if None, prover handles both roles
    judge: Optional[MLXModelConfig] = field(default_factory=lambda: MLXModelConfig(
        model_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
        temperature=0.2,
    ))
    # False = swap models on demand (slower, less RAM)
    keep_both_loaded: bool = True
    verbose: bool = True


class MLXProvider:
    """Local MLX inference provider."""

    def __init__(self, config: MLXConfig):
        self.config = config
        self._models: dict[str, tuple] = {}  # role -> (model, tokenizer, is_base)
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
            )

    def _get_model(self, role: str) -> tuple:
        """Load or retrieve a cached model for the given role ("prover" or "judge")."""
        if role in self._models:
            return self._models[role]

        self._ensure_mlx_lm()

        if role == "judge" and self.config.judge is not None:
            model_config = self.config.judge
        else:
            model_config = self.config.prover

        # Swap models if not keeping both in memory
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

        import os, sys
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # rope_parameters warnings come via print() so they hit stdout;
            # suppress both streams for the duration of the load.
            _devnull = open(os.devnull, "w")
            _old_stdout, _old_stderr = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = _devnull
            try:
                model, tokenizer = self._mlx_lm.load(model_config.model_id)
            finally:
                sys.stdout, sys.stderr = _old_stdout, _old_stderr
                _devnull.close()

        is_base = _is_base_model(model_config.model_id, tokenizer)

        if self.config.verbose:
            kind = "base/completion" if is_base else "instruct"
            print(f"  [MLX] Loaded in {time.time() - t0:.1f}s  [{kind} mode]")

        self._models[role] = (model, tokenizer, is_base)
        return (model, tokenizer, is_base)

    def generate(
        self,
        system: str,
        user: str,
        role: str = "prover",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text using the specified model role."""
        self._ensure_mlx_lm()
        model, tokenizer, is_base = self._get_model(role)

        model_config = (
            self.config.judge if role == "judge" and self.config.judge else self.config.prover
        )
        temp = temperature if temperature is not None else model_config.temperature
        max_tok = max_tokens if max_tokens is not None else model_config.max_tokens

        prompt = _build_prompt(system, user, tokenizer, is_base)

        if self.config.verbose:
            print(f"  [MLX] Generating ({role}, temp={temp}, max_tokens={max_tok})...")
            t0 = time.time()

        from mlx_lm.sample_utils import make_sampler, make_logits_processors
        gen_kwargs = {
            "max_tokens": max_tok,
            "sampler": make_sampler(temp=temp, top_p=model_config.top_p),
            "logits_processors": make_logits_processors(
                repetition_penalty=model_config.repetition_penalty,
            ),
        }
        if model_config.max_kv_size is not None:
            gen_kwargs["max_kv_size"] = model_config.max_kv_size

        # Collect raw token IDs from stream_generate, then bulk-decode.
        # Per-token decode (chunk.text) drops inter-token spaces for
        # tiktoken/SentencePiece models; bulk decode preserves them.
        token_ids = []
        for chunk in self._mlx_lm.stream_generate(
            model, tokenizer,
            prompt=prompt,
            **gen_kwargs,
        ):
            token_ids.append(int(chunk.token))

        token_count = len(token_ids)
        response = _postprocess(tokenizer.decode(token_ids), is_base)

        if self.config.verbose:
            elapsed = time.time() - t0
            tps = token_count / elapsed if elapsed > 0 else 0
            print(f"  [MLX] Generated {token_count} tokens in {elapsed:.1f}s ({tps:.1f} tok/s)")

        return response

    def unload_all(self):
        """Free all loaded models from memory."""
        self._models.clear()
        import gc
        gc.collect()
        if self.config.verbose:
            print("  [MLX] All models unloaded")


class MLXServerProvider:
    """Connect to a local mlx_lm.server OpenAI-compatible API.

    Start with: mlx_lm.server --model <model> --port 8081
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
