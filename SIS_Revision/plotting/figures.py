"""
Figure generation for the SIS revision.

Produces paper-quality figures with large fonts (addressing Reviewer 1 minor comment).
All figures use the surviving-runs prevalence (rho_surviving) as the primary
estimator; rho_naive is shown as a lighter comparison curve where applicable.

Figures generated:
  E1  — m-sweep at fixed τ < 1/λ₁ (prevalence vs m)
  E2  — phase heatmap in (τ, m) space
  E3  — extinction probability heatmap or τ-sweep
  E4  — early-time growth rate r_sim vs r_th
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MultipleLocator, MaxNLocator, AutoMinorLocator

# ── Global plot style ─────────────────────────────────────────────────────────
# Large fonts to address Reviewer 1 minor comment 6
STYLE = {
    "figure.dpi":      300,
    "font.size":       14,
    "axes.labelsize":  16,
    "axes.titlesize":  16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "axes.linewidth":  1.2,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
}


def _apply_style():
    plt.rcParams.update(STYLE)


def savefig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── E1: m-sweep at fixed τ ────────────────────────────────────────────────────

def plot_E1_m_sweep(df: pd.DataFrame, tau_fixed: float, lambda1: float,
                    activity: float, net_name: str, fig_dir: str,
                    correction_factor: float = 1.0):
    """
    Experiment E1: prevalence vs m at a fixed tau below the (corrected) threshold.

    correction_factor: finite-N correction C0 so that the effective threshold is
        tau_c(m) = C0 / (lambda1 + 2*a*m).
    For ER/WS use 1.0 (pure mean-field).  For BA use C0 = tau_c_emp(m=0)*lambda1.
    """
    _apply_style()
    available_taus = df["tau"].unique()
    tau_actual = available_taus[np.argmin(np.abs(available_taus - tau_fixed))]
    if abs(tau_actual - tau_fixed) > 0.02:
        print(f"  [E1] No tau near {tau_fixed:.4f} in {net_name} (closest: {tau_actual:.4f})")
        return
    if abs(tau_actual - tau_fixed) > 1e-4:
        print(f"  [E1] Snapping tau {tau_fixed:.4f} -> {tau_actual:.4f} in {net_name}")
    tau_fixed = tau_actual
    sub = df[np.isclose(df["tau"], tau_fixed, atol=1e-4)].sort_values("m")

    # m_c from corrected threshold: tau * (lambda1 + 2*a*m) > correction_factor
    val = (correction_factor / tau_fixed - lambda1) / (2.0 * activity)
    m_c = max(0, int(np.ceil(val)))

    if correction_factor != 1.0:
        mc_label = (rf"$m_c={m_c}$  (corrected, $C_0={correction_factor:.2f}$)")
    else:
        mc_label = rf"$m_c={m_c}$"

    def _draw(ax):
        ax.plot(sub["m"], sub["rho_surviving"], marker="o", label="surviving-runs avg",
                color="steelblue")
        ax.plot(sub["m"], sub["rho_naive"], marker="s", linestyle="--",
                label="naive avg (all runs)", color="gray", alpha=0.7)
        ax.axvline(m_c, color="crimson", linestyle="--", linewidth=1.5,
                   label=mc_label)
        ax.set_xlabel(r"$m$ (contacts per activation)")
        ax.set_ylabel(r"Steady-state prevalence $\bar\rho$")
        ax.set_title(rf"{net_name} — E1: prevalence vs $m$ at $\tau={tau_fixed:.3f}$")
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.set_xlim(sub["m"].min() - 0.5, sub["m"].max() + 0.5)
        ax.set_ylim(-0.02, min(1.05, sub["rho_surviving"].max() * 1.3 + 0.05))
        ax.legend(fontsize=10)
        ax.grid(which="major", linestyle="-", alpha=0.3)
        ax.grid(which="minor", axis="x", linestyle=":", alpha=0.2)

    path = os.path.join(fig_dir, f"{net_name}_E1_m_sweep.pdf")
    _apply_style(); fig, ax = plt.subplots(figsize=(6.5, 4.5)); _draw(ax)
    savefig(fig, path)
    _apply_style(); fig2, ax2 = plt.subplots(figsize=(6.5, 4.5)); _draw(ax2)
    fig2.tight_layout()
    savefig(fig2, path.replace(".pdf", ".png"))


# ── E2: Phase heatmap (τ, m) ─────────────────────────────────────────────────

def plot_E2_heatmap(df: pd.DataFrame, lambda1: float, activity: float,
                    net_name: str, fig_dir: str, use_surviving: bool = True,
                    correction_factor: float = 1.0):
    """
    Experiment E2: heatmap of prevalence over (τ, m) with theory boundary overlaid.
    Generates two panels: surviving-runs and naive averages side by side.
    """
    _apply_style()
    col = "rho_surviving" if use_surviving else "rho_naive"
    label = "Surviving-runs prevalence" if use_surviving else "Naive prevalence"

    g = (df.groupby(["m", "tau"], as_index=False)
           .agg(rho_surv=("rho_surviving", "mean"),
                rho_naive=("rho_naive", "mean")))

    mat_surv  = g.pivot(index="m", columns="tau", values="rho_surv").sort_index()
    mat_naive = g.pivot(index="m", columns="tau", values="rho_naive").sort_index()

    m_vals   = mat_surv.index.values.astype(float)
    tau_vals = mat_surv.columns.values.astype(float)
    tau_c    = correction_factor / (lambda1 + 2.0 * activity * m_vals)

    if correction_factor != 1.0:
        boundary_label = (rf"$\tau_c = C_0/(\lambda_1+2am)$, "
                          rf"$C_0={correction_factor:.2f}$")
    else:
        boundary_label = r"$\tau_c = 1/(\lambda_1+2am)$"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, mat, title in zip(
        axes,
        [mat_surv.values, mat_naive.values],
        ["Surviving-runs avg", "Naive avg (all runs)"],
    ):
        extent = [tau_vals.min(), tau_vals.max(), m_vals.min(), m_vals.max()]
        im = ax.imshow(mat, origin="lower", aspect="auto", extent=extent,
                       vmin=0, vmax=mat_surv.values[~np.isnan(mat_surv.values)].max(),
                       cmap="viridis")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(r"$\bar\rho$", fontsize=14)

        ax.plot(tau_c, m_vals, "w--", lw=2, label=boundary_label)
        ax.set_xlabel(r"$\tau = \beta/\mu$")
        ax.set_ylabel(r"$m$")
        ax.set_title(f"{net_name} — {title}")
        ax.legend(loc="upper right", fontsize=11)

    fig.suptitle(f"{net_name}  ($\\lambda_1={lambda1:.2f}$, $a={activity}$)",
                 fontsize=15)
    fig.tight_layout()

    suffix = "E2_heatmap"
    savefig(fig, os.path.join(fig_dir, f"{net_name}_{suffix}.png"))


# ── E3: Extinction probability ────────────────────────────────────────────────

def plot_E3_extinction(df: pd.DataFrame, lambda1: float, activity: float,
                       net_name: str, fig_dir: str,
                       correction_factor: float = 1.0):
    """
    Extinction probability heatmap over (τ, m) with theory boundary.
    """
    _apply_style()
    g = (df.groupby(["m", "tau"], as_index=False)
           .agg(ext=("extinction_prob", "mean")))

    mat = g.pivot(index="m", columns="tau", values="ext").sort_index()
    m_vals   = mat.index.values.astype(float)
    tau_vals = mat.columns.values.astype(float)
    tau_c    = correction_factor / (lambda1 + 2.0 * activity * m_vals)

    if correction_factor != 1.0:
        boundary_label = (rf"$\tau_c = C_0/(\lambda_1+2am)$, "
                          rf"$C_0={correction_factor:.2f}$")
    else:
        boundary_label = r"$\tau_c = 1/(\lambda_1+2am)$"

    fig, ax = plt.subplots(figsize=(7, 5))
    extent = [tau_vals.min(), tau_vals.max(), m_vals.min(), m_vals.max()]
    im = ax.imshow(mat.values, origin="lower", aspect="auto", extent=extent,
                   vmin=0, vmax=1, cmap="RdYlGn_r")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Extinction probability", fontsize=14)

    ax.plot(tau_c, m_vals, "b--", lw=2, label=boundary_label)
    ax.set_xlabel(r"$\tau = \beta/\mu$")
    ax.set_ylabel(r"$m$")
    ax.set_title(f"{net_name} — Extinction probability")
    ax.legend(loc="upper right", fontsize=11)
    fig.tight_layout()

    savefig(fig, os.path.join(fig_dir, f"{net_name}_E3_extinction.png"))


# ── E4: Growth rate r_sim vs r_th ────────────────────────────────────────────

def plot_E4_growth_rate(df: pd.DataFrame, lambda1: float, activity: float,
                        net_name: str, fig_dir: str, m_fixed: int = 5,
                        correction_factor: float = 1.0):
    """
    Early-time growth rate: r_sim vs r_th, with corrected tau_c marked.
    """
    _apply_style()
    sub = df[df["m"] == m_fixed].sort_values("tau").copy()
    if sub.empty:
        print(f"  [E4] No data for m={m_fixed} in {net_name}", flush=True)
        return

    tau_c_th = correction_factor / (lambda1 + 2.0 * activity * m_fixed)
    if correction_factor != 1.0:
        tc_label = rf"$\tau_c^{{\rm corr}}={tau_c_th:.3f}$  ($C_0={correction_factor:.2f}$)"
    else:
        tc_label = rf"$\tau_c={tau_c_th:.3f}$"

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    valid_sim = sub.dropna(subset=["r_sim"])
    ax.scatter(valid_sim["tau"], valid_sim["r_sim"], color="steelblue", zorder=5,
               label=r"$r_{\mathrm{sim}}$ (measured)", s=30)
    ax.plot(sub["tau"], sub["r_th"], color="crimson", linestyle="-",
            label=r"$r_{\mathrm{th}} = \mu(\tau(\lambda_1+2am)-1)$")

    ax.axhline(0, color="k", linewidth=0.8, linestyle=":")
    ax.axvline(tau_c_th, color="gray", linestyle="--", linewidth=1.2,
               label=tc_label)

    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("Early-time growth rate $r$")
    ax.set_title(rf"{net_name} — Growth rate at $m={m_fixed}$")
    ax.legend()
    ax.grid(which="major", linestyle="-", alpha=0.3)
    fig.tight_layout()

    savefig(fig, os.path.join(fig_dir, f"{net_name}_E4_growth_rate_m{m_fixed}.png"))


# ── Combined E1+E3 for paper (side-by-side) ───────────────────────────────────

def plot_combined_E1_E3(df: pd.DataFrame, tau_fixed: float, lambda1: float,
                        activity: float, net_name: str, fig_dir: str,
                        correction_factor: float = 1.0):
    """
    Two-panel figure: prevalence vs m (E1) + extinction prob vs m (E3)
    at fixed tau.  Uses the corrected threshold when correction_factor != 1.
    """
    _apply_style()
    available_taus = df["tau"].unique()
    tau_actual = available_taus[np.argmin(np.abs(available_taus - tau_fixed))]
    if abs(tau_actual - tau_fixed) > 0.02:
        return
    tau_fixed = tau_actual
    sub = df[np.isclose(df["tau"], tau_fixed, atol=1e-4)].sort_values("m")
    if sub.empty:
        return

    val = (correction_factor / tau_fixed - lambda1) / (2.0 * activity)
    m_c = max(0, int(np.ceil(val)))
    if correction_factor != 1.0:
        mc_label = rf"$m_c={m_c}$ (corrected, $C_0={correction_factor:.2f}$)"
    else:
        mc_label = rf"$m_c={m_c}$"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: prevalence
    ax1.plot(sub["m"], sub["rho_surviving"], marker="o", color="steelblue",
             label="surviving-runs avg")
    ax1.plot(sub["m"], sub["rho_naive"], marker="s", linestyle="--", color="gray",
             alpha=0.7, label="naive avg")
    ax1.axvline(m_c, color="crimson", linestyle="--", linewidth=1.5,
                label=mc_label)
    ax1.set_xlabel(r"$m$")
    ax1.set_ylabel(r"Steady-state prevalence $\bar\rho$")
    ax1.set_title(rf"(a) Prevalence, $\tau={tau_fixed:.3f}$")
    ax1.legend(fontsize=10)
    ax1.grid(which="major", linestyle="-", alpha=0.3)

    # Right: extinction probability
    ax2.plot(sub["m"], sub["extinction_prob"], marker="^", color="darkorange",
             label="extinction prob")
    ax2.axvline(m_c, color="crimson", linestyle="--", linewidth=1.5,
                label=mc_label)
    ax2.set_xlabel(r"$m$")
    ax2.set_ylabel("Extinction probability")
    ax2.set_title(rf"(b) Extinction prob., $\tau={tau_fixed:.3f}$")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.grid(which="major", linestyle="-", alpha=0.3)

    fig.suptitle(f"{net_name}  ($\\lambda_1={lambda1:.2f}$, $a={activity}$)",
                 fontsize=15)
    fig.tight_layout()
    savefig(fig, os.path.join(fig_dir, f"{net_name}_E1E3_combined.png"))
