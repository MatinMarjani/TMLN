"""
Central configuration for the SIS revision experiments.
All parameter choices are documented here for reproducibility.
"""

# ── Network ──────────────────────────────────────────────────────────────────
N = 10_000          # nodes per network (matches paper)
AVG_DEGREE = 4      # target average degree for all three families

NETWORK_PARAMS = {
    "ER": {"p": AVG_DEGREE / (N - 1)},                 # Erdős–Rényi
    "WS": {"k": AVG_DEGREE, "p_rewire": 0.1},          # Watts–Strogatz
    "BA": {"m0": AVG_DEGREE // 2},                      # Barabási–Albert (m0=2 → avg deg ≈4)
}

GRAPH_DIR = "networks/graphs"   # where adjlists are saved
GRAPH_SEED = 42                 # reproducible network instances

# ── Simulation ────────────────────────────────────────────────────────────────
ACTIVITY = 0.1          # homogeneous activation probability a
MU       = 1.0          # recovery probability per step (τ = β/μ = β)

TIMESTEPS  = 500        # total steps T  (longer than before for cleaner steady state)
T_BURN     = 250        # transient cutoff T_b = T/2
N_RUNS     = 100        # replicates per (m, τ) point
I0_FRAC    = 0.10       # initial infected fraction

# ── Parameter sweeps ─────────────────────────────────────────────────────────
import numpy as np

M_VALUES  = list(range(0, 16))                              # m ∈ {0,…,15}
TAU_VALUES = np.round(np.arange(0.05, 0.35, 0.005), 3).tolist()  # τ sweep

# ── Analysis ──────────────────────────────────────────────────────────────────
EARLY_GROWTH_FRAC = 0.05   # max I/N for early-time growth rate window
EARLY_GROWTH_MIN_POINTS = 10  # minimum time points needed to fit the slope

# ── Output ────────────────────────────────────────────────────────────────────
DATA_DIR = "data"       # CSV outputs
FIG_DIR  = "figures"   # plot outputs
