#!/usr/bin/env bash
# One-time setup for the Lean 4 verifier: installs Lean (via elan) and creates a
# Lake project with Mathlib, downloading Mathlib's pre-built cache.
# Needs ~5 GB of disk. On a cluster, run it on every machine with the same path.
#
#   bash scripts/setup_lean.sh            # creates ./lean_workspace
#   bash scripts/setup_lean.sh /data/lean # or a directory of your choice
set -euo pipefail

DIR="${1:-lean_workspace}"

if ! command -v elan >/dev/null 2>&1; then
  echo ">> Installing elan (the Lean version manager)"
  curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
    | sh -s -- -y --default-toolchain none
  # shellcheck disable=SC1091
  source "$HOME/.elan/env"
fi

if [ ! -d "$DIR" ]; then
  echo ">> Creating a Mathlib project in $DIR"
  lake +leanprover-community/mathlib4:lean-toolchain new "$DIR" math
fi

cd "$DIR"
echo ">> Downloading pre-built Mathlib (this is the slow part)"
lake exe cache get
lake build

echo ">> Smoke test"
cat > .mf_smoke.lean <<'EOF'
import Mathlib
theorem mf_smoke (a b : ℝ) : 2 * a * b ≤ a ^ 2 + b ^ 2 := by nlinarith [sq_nonneg (a - b)]
#print axioms mf_smoke
EOF
lake env lean .mf_smoke.lean
rm -f .mf_smoke.lean
echo ">> Lean + Mathlib ready in $(pwd). Point engines.selfplay.lean.project_dir here."
