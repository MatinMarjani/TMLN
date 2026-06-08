"""
Read results CSVs and generate all paper figures.

Usage (from SIS_Revision/ with venv active):
    python scripts/make_figures.py

Produces figures in figures/<NETWORK>_*.png for each network.

You can control which τ to use for E1/E3 m-sweep and which m to use for E4
growth-rate plot via the TAU_FIXED and M_FIXED dicts below (one per network).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import config
from networks.generate import generate_networks, spectral_radius
from plotting.figures import (
    plot_E1_m_sweep,
    plot_E2_heatmap,
    plot_E3_extinction,
    plot_E4_growth_rate,
    plot_combined_E1_E3,
)

FIG_DIR = "figures"

# τ values used for the m-sweep (E1/E3): choose one below the static τ_c
# (override here after you know λ₁ from your networks)
TAU_FIXED = {
    "ER": 0.165,   # just below empirical tau_c(m=0)~0.200; gives m_c~5-7
    "WS": 0.195,   # just below empirical tau_c(m=0)~0.280; gives m_c~5-7
    "BA": 0.060,   # well below empirical tau_c(m=0)~0.087; corrected m_c=37 (uses zoom data)
}

# m values used for the growth-rate plot (E4)
M_FIXED = {
    "ER": 5,
    "WS": 5,
    "BA": 5,
}

# Finite-N correction factor C0 = tau_c_empirical(m=0) * lambda1.
# For ER/WS pure mean-field is accurate so C0=1.0.
# For BA (gamma=3): the effective threshold has a finite-N offset;
# C0 is calibrated from the empirically observed static onset ~0.087.
# Reference: Castellano & Pastor-Satorras, Phys. Rev. X 10, 011070 (2020).
CORRECTION_FACTOR = {
    "ER": 1.0,
    "WS": 1.0,
    "BA": 1.4236,   # = 0.087 * 16.3635
}


def main():
    # Load networks just for λ₁ (no re-simulation needed)
    print("Loading networks for lambda1 values...")
    graphs = generate_networks(
        config.N,
        config.NETWORK_PARAMS,
        config.GRAPH_DIR,
        seed=config.GRAPH_SEED,
    )

    for net in ["ER", "WS", "BA"]:
        csv_path = os.path.join(config.DATA_DIR, f"{net}_results.csv")
        if not os.path.exists(csv_path):
            print(f"  [{net}] No results CSV found at {csv_path} — run run_sweep.py first.")
            continue

        print(f"\n{'='*50}")
        print(f" {net}")
        print(f"{'='*50}")

        df = pd.read_csv(csv_path)
        lambda1 = float(df["lambda1"].iloc[0])
        activity = float(df["activity"].iloc[0])

        print(f"  lambda1 = {lambda1:.4f}   tau_c(m=0) = {1/lambda1:.4f}")

        tau_fixed = TAU_FIXED[net]
        m_fixed   = M_FIXED[net]
        cf        = CORRECTION_FACTOR[net]
        fig_dir   = os.path.join(FIG_DIR, net)

        # For BA m-sweep plots (E1, E3, combined): use zoom data (m=0..60, tau=0.060)
        # so m_c=37 is visible. E2 and E4 still use the full grid (BA_results.csv).
        if net == "BA":
            zoom_path = os.path.join(config.DATA_DIR, "BA_zoom.csv")
            if os.path.exists(zoom_path):
                df_msweep = pd.read_csv(zoom_path)
                print(f"  Using BA_zoom.csv for m-sweep plots (m=0..60, tau={tau_fixed})")
            else:
                print(f"  BA_zoom.csv not found, falling back to BA_results.csv")
                df_msweep = df
        else:
            df_msweep = df

        plot_E1_m_sweep(df_msweep, tau_fixed, lambda1, activity, net, fig_dir,
                        correction_factor=cf)
        plot_E2_heatmap(df, lambda1, activity, net, fig_dir,
                        correction_factor=cf)
        plot_E3_extinction(df, lambda1, activity, net, fig_dir,
                           correction_factor=cf)
        if net != "BA":   # BA skipped: mean-field r_th grossly overestimates on scale-free nets
            plot_E4_growth_rate(df, lambda1, activity, net, fig_dir,
                                m_fixed=m_fixed, correction_factor=cf)
        plot_combined_E1_E3(df_msweep, tau_fixed, lambda1, activity, net, fig_dir,
                            correction_factor=cf)

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
