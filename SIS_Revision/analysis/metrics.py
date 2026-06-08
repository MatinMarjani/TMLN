"""
Post-simulation metrics for the SIS revision.

Implements the three estimators that address reviewer comments:

1. rho_naive      — Eq. (steady-state) from paper: simple time-and-run average
                    after burn-in (includes extinct runs with I=0).

2. rho_surviving  — Jensen & Dickman (1993) surviving-runs average:
                    at each time t, average only over replicates where I_t > 0.
                    This is the standard estimator near absorbing-state transitions.

3. extinction_prob — fraction of replicates ending with I_T = 0.

4. r_sim          — early-time log-linear growth rate (slope of log I_t vs t
                    while mean prevalence stays below EARLY_GROWTH_FRAC).
                    Compared against the theoretical near-DFE rate
                    r_th = μ(τ(λ₁ + 2am) - 1).
"""

import numpy as np


def compute_metrics(
    I_series: np.ndarray,
    N: int,
    T_burn: int,
    early_growth_frac: float = 0.05,
    early_growth_min_points: int = 10,
) -> dict:
    """
    Compute all metrics from raw simulation output.

    Parameters
    ----------
    I_series : np.ndarray, shape (n_runs, T)
        Infected count at each timestep for every replicate.
    N : int
        Number of nodes.
    T_burn : int
        Burn-in (transient) cutoff; only steps t >= T_burn used for steady-state.
    early_growth_frac : float
        Maximum mean prevalence (I/N) to be included in the early-growth window.
    early_growth_min_points : int
        Minimum number of time points required for a valid growth-rate fit.

    Returns
    -------
    dict with keys:
        rho_naive       (float) — naive steady-state prevalence (all runs)
        rho_surviving   (float) — surviving-runs steady-state prevalence
        extinction_prob (float) — fraction of runs extinct at t=T
        r_sim           (float or NaN) — estimated early-time growth rate
        n_surviving_ss  (float) — mean number of surviving runs over [T_burn, T]
    """
    n_runs, T = I_series.shape
    rho = I_series / N  # (n_runs, T) — prevalence time series

    # ── 1. Extinction probability ─────────────────────────────────────────────
    extinction_prob = float((I_series[:, -1] == 0).mean())

    # ── 2. Naive steady-state prevalence ─────────────────────────────────────
    # Average over all replicates and all post-burn-in steps (includes zeros
    # from extinct runs, per original paper code).
    rho_post = rho[:, T_burn:]                      # (n_runs, T-T_burn)
    rho_naive = float(rho_post.mean())

    # ── 3. Surviving-runs steady-state prevalence ─────────────────────────────
    # Jensen & Dickman (1993): at each time t, condition on I_t > 0.
    # We compute rho_surviving(t) = mean of I_t/N over {r : I_t^r > 0}, then
    # average over t in [T_burn, T].
    alive_post = (I_series[:, T_burn:] > 0)         # bool (n_runs, T-T_burn)
    n_surviving_per_t = alive_post.sum(axis=0)       # (T-T_burn,)

    # Masked mean: sum of rho over surviving runs / count of surviving runs
    rho_post_alive = np.where(alive_post, rho_post, 0.0)
    # Avoid division by zero for time steps where all runs are extinct
    with np.errstate(invalid="ignore"):
        rho_surv_t = np.where(
            n_surviving_per_t > 0,
            rho_post_alive.sum(axis=0) / n_surviving_per_t,
            np.nan,
        )
    rho_surviving = float(np.nanmean(rho_surv_t)) if np.any(~np.isnan(rho_surv_t)) else 0.0
    n_surviving_ss = float(np.nanmean(n_surviving_per_t)) if np.any(n_surviving_per_t > 0) else 0.0

    # ── 4. Early-time growth rate ─────────────────────────────────────────────
    # Fit slope of log(mean_I_t) vs t over the early fixed window [0, T_burn/2].
    # We use the mean over *surviving* runs to avoid extinct-run zeros pulling
    # the log fit toward -inf.  Only include steps where at least one run is alive.
    early_end = min(T_burn // 2, T)                 # first half of burn-in
    mean_I_surv = np.where(
        n_surviving_per_t > 0,
        rho_post_alive.sum(axis=0) / np.maximum(n_surviving_per_t, 1),
        np.nan,
    ) * N                                           # back to counts

    # Use all steps from 0 to early_end (pre-burn-in)
    full_surv_counts = np.full(T, np.nan)
    # surviving mean counts over the full time series (not just post-burn)
    alive_full = (I_series > 0)                     # (n_runs, T)
    n_surv_full = alive_full.sum(axis=0)            # (T,)
    I_masked    = np.where(alive_full, I_series, 0)
    with np.errstate(invalid="ignore"):
        mean_I_surv_full = np.where(
            n_surv_full > 0,
            I_masked.sum(axis=0) / np.maximum(n_surv_full, 1),
            np.nan,
        )

    early_t = np.arange(early_end)
    valid = early_t[~np.isnan(mean_I_surv_full[early_t])
                    & (mean_I_surv_full[early_t] > 0)]

    r_sim = np.nan
    if len(valid) >= early_growth_min_points:
        log_counts = np.log(mean_I_surv_full[valid])
        slope, _ = np.polyfit(valid, log_counts, 1)
        r_sim = float(slope)

    return {
        "rho_naive": rho_naive,
        "rho_surviving": rho_surviving,
        "extinction_prob": extinction_prob,
        "r_sim": r_sim,
        "n_surviving_ss": n_surviving_ss,
    }


def theoretical_growth_rate(mu: float, tau: float, lambda1: float, a: float, m: int) -> float:
    """
    Near-DFE theoretical early-time growth rate.

    r_th = μ(τ(λ₁ + 2am) - 1)

    Positive ↔ supercritical (infection grows); negative ↔ subcritical (dies out).
    """
    return mu * (tau * (lambda1 + 2 * a * m) - 1)


def theoretical_threshold(lambda1: float, a: float, m: int) -> float:
    """τ_c = 1 / (λ₁ + 2am)"""
    return 1.0 / (lambda1 + 2.0 * a * m)


def critical_m(tau: float, lambda1: float, a: float) -> int:
    """m_c = max(0, ceil((1/τ - λ₁) / (2a)))"""
    val = (1.0 / tau - lambda1) / (2.0 * a)
    return max(0, int(np.ceil(val)))
