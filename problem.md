# Navier-Stokes Existence and Smoothness

## Problem Statement

Let $u: \mathbb{R}^3 \times [0, \infty) \to \mathbb{R}^3$ be a velocity field and
$p: \mathbb{R}^3 \times [0, \infty) \to \mathbb{R}$ be a pressure field satisfying
the incompressible Navier-Stokes equations:

$$
\partial_t u + (u \cdot \nabla) u = \nu \Delta u - \nabla p + f
$$
$$
\nabla \cdot u = 0
$$

with initial data $u(x, 0) = u_0(x)$ where $u_0 \in C^\infty(\mathbb{R}^3)$ is
divergence-free and $f \equiv 0$ (unforced case).

**Goal**: Prove that for any such smooth, divergence-free initial data with
$|D^\alpha u_0(x)| \leq C_{\alpha K}(1 + |x|)^{-K}$ for all $\alpha, K$,
there exists a smooth solution $(u, p)$ defined for all $t \geq 0$ satisfying
$u \in C^\infty(\mathbb{R}^3 \times [0, \infty))$.

## Proof Tree — Root Decomposition

The top-level goal decomposes into these major sub-goals. The agent should attempt
them in dependency order, but may explore any leaf at any time.

### Branch A: Local Existence (KNOWN — use as foundation)
- **A.1** Local-in-time existence of mild solutions via Fujita-Kato
- **A.2** Regularity of mild solutions on existence interval
- **A.3** Continuation criterion: solution extends as long as $\|u(\cdot, t)\|_{L^3}$ stays finite
- **Status**: ESTABLISHED. These are known results. The agent should formalize them
  as lemmas and build upon them.

### Branch B: A Priori Energy Estimates
- **B.1** Derive the energy inequality: $\|u(t)\|_{L^2}^2 + 2\nu \int_0^t \|\nabla u\|_{L^2}^2 \leq \|u_0\|_{L^2}^2$
- **B.2** Higher-order energy estimates (bootstrap from $H^1$ to $H^k$)
- **B.3** Establish that $L^2$ energy controls do NOT by themselves prevent blow-up
  (this is the fundamental gap)
- **Depends on**: A.1

### Branch C: Blow-Up Exclusion Strategies
This is the critical branch. Multiple sub-strategies exist:

#### C.1: Scaling Analysis
- Show that any hypothetical blow-up has self-similar structure
- Derive constraints on self-similar profiles
- Attempt to show no non-trivial self-similar blow-up exists in $L^3$

#### C.2: Regularity via Critical Norms
- **C.2.1** Escauriaza-Seregin-Šverák: blow-up implies $\|u(t)\|_{L^{3,\infty}} \to \infty$
- **C.2.2** Attempt to strengthen to: blow-up implies $\|u(t)\|_{L^3} \to \infty$
- **C.2.3** Show this contradicts energy estimates (THIS IS THE GAP — may not close)

#### C.3: Geometric / Topological Constraints
- Analyze vorticity $\omega = \nabla \times u$ evolution
- Vorticity equation: $\partial_t \omega + (u \cdot \nabla)\omega = (\omega \cdot \nabla)u + \nu \Delta \omega$
- **C.3.1** Establish BKM criterion: blow-up iff $\int_0^{T^*} \|\omega\|_{L^\infty} dt = \infty$
- **C.3.2** Attempt to bound vorticity growth using geometric properties of vortex tubes
- **C.3.3** Explore Constantin-Fefferman-Majda direction coherence condition

#### C.4: Frequency-Space / Littlewood-Paley Approach
- **C.4.1** Decompose solution into frequency shells
- **C.4.2** Establish energy flux bounds between shells
- **C.4.3** Attempt to show cascade cannot concentrate at infinite frequency in finite time

#### C.5: Novel Hypotheses (Agent-Generated)
- The agent may propose entirely new approaches
- Each must be decomposed into verifiable sub-steps
- Reward is proportional to how far the sub-proof progresses before failing

## Known Obstacles

The agent should be aware of these fundamental difficulties:

1. **Supercritical scaling**: NS in 3D is energy-supercritical. The energy norm $L^2$
   is subcritical, so energy bounds alone cannot control the solution. The critical
   space is $\dot{H}^{1/2}$ or $L^3$.

2. **Nonlinear term structure**: $(u \cdot \nabla)u$ transfers energy across scales.
   Unlike 2D where vortex stretching is absent, 3D allows $\omega \cdot \nabla u$
   to amplify vorticity without bound (a priori).

3. **No known monotone quantity**: There is no known quantity that is both controlled
   by initial data AND controls regularity in 3D.

## Evaluation Criteria

For each proof attempt at a node, the verifier scores:

- **VERIFIED**: Lean 4 type-checks OR LLM-judge confirms logical validity (score = 1.0)
- **PARTIAL**: Proof has correct structure but gaps remain (score = 0.3-0.7)
- **FAILED**: Logical error, circular reasoning, or unjustified step (score = 0.0)
- **NOVEL**: Introduces a genuinely new technique or connection (bonus = +0.2)

## Agent Instructions

You are a research mathematician exploring proof strategies for the Navier-Stokes
existence and smoothness problem. At each step:

1. Examine the current proof tree state (provided as JSON)
2. Select the most promising unexplored node (or propose a new branch)
3. Generate a rigorous proof attempt for that node
4. If your attempt fails verification, analyze WHY and suggest which alternative
   branch to explore next
5. If you discover a connection between branches, propose merging them

Be precise. Be rigorous. Label every assumption. If a step requires an unproven
lemma, create a child node for it rather than assuming it.
