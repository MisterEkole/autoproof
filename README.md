# autoproof

The idea behind this project is pretty simple: give an AI agent an unsolved math problem, let it explore proof strategies like a researcher would — decomposing, attempting, failing, backtracking — and see how far it gets.

It uses Monte Carlo Tree Search to navigate the proof space. Each node in the tree is a sub-goal. The agent picks the most promising one, generates a proof attempt, gets it scored, and propagates the result back up the tree. Dead ends get pruned. Good branches get explored deeper.

The default problem is Navier-Stokes global existence, but the framework is problem-agnostic. To use it on any math or physics problem, just rewrite `problem.md` with your theorem, its known sub-goals, and the proof strategies worth exploring. The agent handles the rest.

## How it works

```
select node (UCB1) → generate proof → verify (Lean 4 or LLM judge) → backpropagate → repeat
```

1. **Select** — UCB1 picks the most promising unvisited node. Unvisited nodes go first (score = ∞).
2. **Attempt** — LLM generates a proof for the selected sub-goal.
3. **Verify** — Lean 4 if available, otherwise an LLM judge scores it 0–1.
4. **Update** — score propagates up the tree with geometric decay. Nodes with 3+ consecutive failures get pruned.
5. **Repeat** — until budget exhausted or proof found.

## Quick start

```bash
# API mode
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
python orchestrator.py --config config.online.toml --problem problem.md

# Offline mode (Apple Silicon)
pip install mlx-lm
python orchestrator.py --config config.offline.toml --problem problem.md

# View the proof tree live
python serve_viewer.py   # opens http://localhost:7331
```

## Project structure

```
orchestrator.py       main loop
proof_tree.py         MCTS tree with UCB1 selection and backpropagation
verifier.py           Lean 4 + LLM-judge verification
llm_interface.py      provider abstraction (Anthropic, OpenAI, MLX)
mlx_provider.py       local inference on Apple Silicon
config.py             config dataclasses
config.online.toml    API mode config
config.offline.toml   offline MLX config
problem.md            problem spec and proof decomposition
viewer.html           proof tree viewer (served by serve_viewer.py)
serve_viewer.py       tiny HTTP server for the viewer
logs/                 tree snapshots saved every N iterations
```

## Offline mode (Apple Silicon)

Runs fully on-device via [mlx-lm](https://github.com/ml-explore/mlx-lm). No API key needed.

| Role | Model | Size |
|------|-------|------|
| Prover | `mlx-community/Qwen2.5-Math-7B-Instruct-4bit` | ~4GB |
| Judge | `mlx-community/Qwen2.5-7B-Instruct-4bit` | ~4GB |

Override the judge model via `export AUTOPROOF_JUDGE_MODEL="<model-id>"`. Set `AUTOPROOF_DUAL_LOADED=0` if RAM is tight.

## What to expect

- Foundational nodes (local existence, energy estimates) — should reach `verified` with Sonnet+, `partial` with Haiku.
- Open problem branches (blow-up exclusion) — will plateau at partial or fail. That's correct behavior.

## Credit

Adapted from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch), which applies the same loop to ML training runs. The core idea is the same — automated exploration with a feedback signal — just applied to proof search instead of hyperparameter search.

## License

MIT
