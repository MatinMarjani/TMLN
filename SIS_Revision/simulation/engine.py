"""
Vectorized discrete-time SIS simulation on a static + activity-driven multiplex.

Algorithm (synchronous update, matches Algorithm 1 in the paper):
  For each step t:
    1. Build temporal graph G_t: each node activates with prob a and picks m
       uniform random targets (undirected, one-step lifetime).
    2. Compute infection pressure from BOTH layers simultaneously.
    3. Apply recoveries and infections simultaneously (synchronous).

Performance strategy
--------------------
States are stored as a (N, n_runs) float32 matrix so that the dominant cost
(static-layer pressure) is ONE sparse-matrix × dense-matrix product per step
(BLAS-level, cache-friendly) rather than n_runs separate sparse mat-vec calls.

Temporal layer: sampled independently per run inside the T-length loop but
uses fast numpy operations (integers(), np.add.at).  The expected number of
active nodes per run is N*a (e.g. 10000*0.1 = 1000) and the inner work is
O(N*a*m) per run per step — manageable.

Returns the full I_t time series for every replicate so that downstream
analysis can compute both naive and surviving-runs averages.
"""

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix


def _sparse_adj(G):
    return nx.to_scipy_sparse_array(G, format="csr", dtype=np.float32)


def run_sis(
    G_static,
    a: float,
    m: int,
    beta: float,
    mu: float,
    T: int,
    n_runs: int,
    I0_frac: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Run SIS simulations on a static + activity-driven multiplex.

    Parameters
    ----------
    G_static : networkx.Graph  or  scipy CSR matrix (N×N)
    a        : float  — activation probability
    m        : int    — contacts per activation
    beta     : float  — per-contact infection probability
    mu       : float  — per-step recovery probability
    T        : int    — timesteps per replicate
    n_runs   : int    — number of independent replicates
    I0_frac  : float  — initial infected fraction
    seed     : int | None

    Returns
    -------
    I_series : np.ndarray, shape (n_runs, T), dtype int32
    """
    if isinstance(G_static, (csr_matrix,)) or hasattr(G_static, "toarray"):
        A = G_static.astype(np.float32)
        N = A.shape[0]
    else:
        A = _sparse_adj(G_static)
        N = G_static.number_of_nodes()

    rng = np.random.default_rng(seed)
    log1mb = np.float32(np.log1p(-beta))   # log(1-β), precomputed

    # ── Batch all runs together: states[i, r] = infection state of node i in run r
    # Shape: (N, n_runs), float32  (1=I, 0=S)
    states = (rng.random((N, n_runs)) < I0_frac).astype(np.float32)
    I_series = np.zeros((n_runs, T), dtype=np.int32)

    for t in range(T):
        # ── Static layer pressure ────────────────────────────────────────────
        # One batched sparse-matrix × dense-matrix product for ALL runs.
        # Result shape: (N, n_runs)
        static_pressure = A @ states   # BLAS-level, ~10-50× faster than n_runs mat-vecs

        # ── Temporal layer pressure ──────────────────────────────────────────
        # temp_pressure[i, r] = number of infected temporal contacts of node i in run r
        temp_pressure = np.zeros((N, n_runs), dtype=np.float32)

        if m > 0:
            for r in range(n_runs):
                s_r = states[:, r]                   # (N,) — current run's states
                active_idx = np.where(rng.random(N) < a)[0]
                n_active = len(active_idx)
                if n_active == 0:
                    continue

                # Sample m targets per active node from {0..N-1}\{i}
                raw = rng.integers(0, N - 1, size=(n_active, m))
                targets = np.where(raw >= active_idx[:, None], raw + 1, raw)

                src = np.repeat(active_idx, m)   # (n_active*m,)
                dst = targets.ravel()            # (n_active*m,)

                # Undirected: pressure on dst from infected src, and vice-versa
                np.add.at(temp_pressure[:, r], dst, s_r[src])
                np.add.at(temp_pressure[:, r], src, s_r[dst])

        # ── State transitions ────────────────────────────────────────────────
        total_pressure = static_pressure + temp_pressure  # (N, n_runs)

        susceptible = 1.0 - states   # (N, n_runs)

        # P(infection) = 1-(1-β)^k  computed via log for numerical stability
        p_infect = -np.expm1(total_pressure * log1mb)  # (N, n_runs)
        new_infected = susceptible * (rng.random((N, n_runs)) < p_infect)

        # P(recovery) = μ
        new_recovered = states * (rng.random((N, n_runs)) < mu)

        # Synchronous update  (clip to [0,1] to handle any float rounding)
        states = np.clip(states + new_infected - new_recovered, 0.0, 1.0)

        I_series[:, t] = states.sum(axis=0).astype(np.int32)

    return I_series
