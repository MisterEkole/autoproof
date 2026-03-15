#!/usr/bin/env python3
"""
Set up a Lean 4 project with Mathlib for formal verification.

This is the analog of autoresearch's prepare.py — it sets up the
fixed infrastructure that the agent doesn't modify.

Run once:
    python setup_lean.py

Requires: Lean 4 + elan installed
    curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
"""

import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path("lean_project")
PROJECT_NAME = "AutoProof"


def run(cmd: str, cwd: Path | None = None, check: bool = True):
    print(f"  $ {cmd}")
    result = subprocess.run(
        cmd, shell=True, cwd=cwd,
        capture_output=True, text=True
    )
    if result.stdout.strip():
        print(f"    {result.stdout.strip()[:200]}")
    if result.returncode != 0 and check:
        print(f"  ERROR: {result.stderr.strip()[:500]}")
        sys.exit(1)
    return result


def setup():
    print("="*50)
    print("  autoproof — Lean 4 project setup")
    print("="*50)

    # Check Lean is installed
    result = run("lean --version", check=False)
    if result.returncode != 0:
        print("\nLean 4 not found. Install with:")
        print("  curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh")
        print("\nYou can still use autoproof with LLM-judge (set use_lean4=false in config.toml)")
        sys.exit(0)

    # Create Lake project
    if PROJECT_DIR.exists():
        print(f"\nProject directory {PROJECT_DIR} already exists. Skipping.")
        return

    print(f"\nCreating Lean 4 project at {PROJECT_DIR}...")
    run(f"lake init {PROJECT_NAME} math", cwd=Path("."))

    # The project is created in the current directory, move it
    if Path(PROJECT_NAME).exists() and not PROJECT_DIR.exists():
        Path(PROJECT_NAME).rename(PROJECT_DIR)

    # Create definitions file for NS-specific types
    defs_dir = PROJECT_DIR / PROJECT_NAME
    defs_dir.mkdir(parents=True, exist_ok=True)

    defs_file = defs_dir / "Defs.lean"
    defs_file.write_text("""/-
  AutoProof — Navier-Stokes definitions and common lemmas.

  This file contains the basic type definitions and known results
  that proof attempts can build upon. Do NOT modify during runs.
-/
import Mathlib

-- Placeholder: NS-specific definitions will be added as the
-- Mathlib formalization of fluid dynamics matures.
-- For now, proof attempts should define what they need inline.

/-- Marker for sub-goals that are known results -/
axiom known_result : Prop

/-- Marker for sub-goals that need proof -/
axiom needs_proof : Prop
""")

    print(f"\nDefs file created: {defs_file}")

    # Fetch Mathlib (this takes a while)
    print("\nFetching Mathlib dependencies (this may take several minutes)...")
    run("lake update", cwd=PROJECT_DIR)
    run("lake exe cache get", cwd=PROJECT_DIR)

    print("\n" + "="*50)
    print("  Lean 4 project ready!")
    print(f"  Project: {PROJECT_DIR}")
    print(f"  Set use_lean4=true in config.toml to enable formal verification")
    print("="*50)


if __name__ == "__main__":
    setup()
