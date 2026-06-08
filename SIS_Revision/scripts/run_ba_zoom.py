"""
Targeted sweep: BA-10000 at tau=0.060, m=0..60.

tau=0.060 is just below the static threshold tau_c(m=0)=0.0611,
so theory says m_c=2.  We scan to m=60 to find the empirical
transition (expected around m~48-50 based on probes).

Reuses existing m=0..15 data from BA_results.csv if present.
New results are appended to BA_zoom.csv.
"""

import sys, os, time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from simulation.engine import run_sis
from analysis.metrics import (compute_metrics, theoretical_growth_rate,
                               theoretical_threshold)


TAU      = 0.060
M_VALUES = list(range(0, 61))     # 0 to 60
N_RUNS   = 100
T        = 500
T_BURN   = 250
MU       = 1.0
ACTIVITY = 0.1
N        = 10_000
LAMBDA1  = 16.3635               # from existing run

GRAPH_PATH = os.path.abspath("networks/graphs/BA-10000.adjlist")
CSV_OUT    = os.path.join(config.DATA_DIR, "BA_zoom.csv")
N_WORKERS  = max(1, os.cpu_count() // 2)


def _worker(args):
    graph_path, m, tau, mu, T, n_runs, I0_frac, T_burn, activity, lambda1, seed = args
    G    = nx.read_adjlist(graph_path, nodetype=int)
    beta = tau * mu
    I_series = run_sis(G, a=activity, m=m, beta=beta, mu=mu,
                       T=T, n_runs=n_runs, I0_frac=I0_frac, seed=seed)
    met   = compute_metrics(I_series, N=N, T_burn=T_burn)
    r_th  = theoretical_growth_rate(mu, tau, lambda1, activity, m)
    tau_c = theoretical_threshold(lambda1, activity, m)
    r_sim = met["r_sim"]
    return {
        "network": "BA", "m": m, "tau": tau,
        "beta": round(beta, 6), "mu": mu, "activity": activity,
        "lambda1": round(lambda1, 6), "tau_c_mf": round(tau_c, 6),
        "rho_naive":      round(met["rho_naive"],      6),
        "rho_surviving":  round(met["rho_surviving"],  6),
        "extinction_prob":round(met["extinction_prob"], 4),
        "r_sim": round(r_sim, 6) if not np.isnan(r_sim) else np.nan,
        "r_th":  round(r_th, 6),
        "n_surviving_ss": round(met["n_surviving_ss"], 2),
        "n_runs": n_runs, "T": T, "T_burn": T_burn,
    }


def main():
    os.makedirs(config.DATA_DIR, exist_ok=True)

    # Load existing data (m=0..15 already computed)
    existing_rows = []
    done_m = set()
    if os.path.exists(CSV_OUT):
        df_ex = pd.read_csv(CSV_OUT)
        existing_rows = df_ex.to_dict("records")
        done_m = set(df_ex["m"].astype(int).tolist())
        print(f"Resuming: {len(done_m)} m values already done.")

    # Also pull from the main BA_results.csv if available
    main_csv = os.path.join(config.DATA_DIR, "BA_results.csv")
    if os.path.exists(main_csv):
        df_main = pd.read_csv(main_csv)
        df_tau  = df_main[np.isclose(df_main["tau"], TAU, atol=0.001)]
        for _, row in df_tau.iterrows():
            m_val = int(row["m"])
            if m_val not in done_m:
                existing_rows.append(row.to_dict())
                done_m.add(m_val)
        print(f"Loaded {len(df_tau)} rows from BA_results.csv at tau={TAU}")

    tasks = []
    for idx, m in enumerate(M_VALUES):
        if m in done_m:
            continue
        seed = 42 + idx * 997
        tasks.append((GRAPH_PATH, m, TAU, MU, T, N_RUNS, config.I0_FRAC,
                      T_BURN, ACTIVITY, LAMBDA1, seed))

    print(f"\nBA zoom sweep: tau={TAU}, m=0..60, workers={N_WORKERS}")
    print(f"Tasks remaining: {len(tasks)} / {len(M_VALUES)}")
    if not tasks:
        print("Nothing to do — all done.")
        _save_and_plot(existing_rows)
        return

    # Probe timing
    print("Probing one point...")
    t0 = time.perf_counter()
    _worker(tasks[0])
    probe_t = time.perf_counter() - t0
    est = probe_t * len(tasks) / N_WORKERS
    print(f"  {probe_t:.1f}s/point  => est. {est/60:.0f} min with {N_WORKERS} workers\n")

    rows = list(existing_rows)
    done_m_set = set(done_m)

    t_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_worker, t): t for t in tasks[1:]}

        # Handle probe result
        r = _worker(tasks[0])
        r["network"] = "BA"
        rows.append(r)
        done_m_set.add(r["m"])
        pd.DataFrame(rows).to_csv(CSV_OUT, index=False)

        for future in as_completed(futures):
            try:
                r = future.result()
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
            r["network"] = "BA"
            rows.append(r)
            done_m_set.add(r["m"])

            elapsed = time.perf_counter() - t_start
            n_done  = len(done_m_set) - len(done_m)
            rate    = n_done / elapsed if elapsed > 0 else 1e-9
            left    = (len(tasks) - n_done) / rate / 60

            r_sim = r["r_sim"]
            rs_str = f"{r_sim:.4f}" if isinstance(r_sim,float) and not np.isnan(r_sim) else "  nan "
            print(f"  m={r['m']:2d}  tau_c={r['tau_c_mf']:.4f}"
                  f"  surv={r['rho_surviving']:.4f}"
                  f"  ext={r['extinction_prob']:.2f}"
                  f"  [{left:.0f}min left]")

            pd.DataFrame(rows).to_csv(CSV_OUT, index=False)

    _save_and_plot(rows)


def _save_and_plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(rows)
    df = df[np.isclose(df["tau"].astype(float), TAU, atol=0.001)].sort_values("m")

    # Pure mean-field m_c  (tau*(lam1+2am) > 1)
    mc_theory = max(0, int(np.ceil((1/TAU - LAMBDA1) / (2*ACTIVITY))))

    # Finite-N corrected m_c using empirical static threshold
    # C0 = tau_c_empirical(m=0) * lambda_1  ≈ 1.42
    # tau_c_empirical(m=0) estimated from simulation: ~0.087
    # (extinction_prob first dips below 1.0 around tau=0.085-0.090)
    TAU_C_EMP_M0 = 0.087          # measured from BA data
    C0 = TAU_C_EMP_M0 * LAMBDA1   # finite-N correction factor ~1.42
    mc_corrected = max(0, int(np.ceil((C0/TAU - LAMBDA1) / (2*ACTIVITY))))

    # Empirical m_c from zoom data: where extinction_prob first drops below 0.5
    below = df[df["extinction_prob"] < 0.5]
    mc_emp = int(below["m"].iloc[0]) if len(below) > 0 else None

    print(f"C0 (finite-N correction) = {C0:.4f}")
    print(f"m_c (pure mean-field)    = {mc_theory}")
    print(f"m_c (corrected, C0)      = {mc_corrected}")
    print(f"m_c (empirical, ext<0.5) = {mc_emp}")

    plt.rcParams.update({"font.size": 14, "axes.labelsize": 16,
                         "axes.titlesize": 15, "figure.dpi": 200})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for ax in (ax1, ax2):
        # Corrected m_c — only theoretical prediction shown (same style as ER/WS)
        ax.axvline(mc_corrected, color="crimson", ls="--", lw=2,
                   label=rf"$m_c={mc_corrected}$  "
                         rf"($\tau(\lambda_1+2am)>C_0$, $C_0={C0:.2f}$)")

    # Prevalence
    ax1.plot(df["m"], df["rho_surviving"], "o-", color="steelblue",
             label="surviving-runs avg")
    ax1.plot(df["m"], df["rho_naive"],    "s--", color="gray", alpha=0.7,
             label="naive avg")
    ax1.set_xlabel(r"$m$ (temporal contacts per activation)")
    ax1.set_ylabel(r"Steady-state prevalence $\bar\rho$")
    ax1.set_title(rf"BA — prevalence vs $m$ at $\tau={TAU}$")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Extinction probability
    ax2.plot(df["m"], df["extinction_prob"], "^-", color="darkorange",
             label="extinction probability")
    ax2.axhline(0.5, color="k", ls=":", lw=0.8, alpha=0.4)
    ax2.set_xlabel(r"$m$ (temporal contacts per activation)")
    ax2.set_ylabel("Extinction probability")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title(rf"BA — extinction prob vs $m$ at $\tau={TAU}$")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    fig.suptitle(
        rf"BA ($\lambda_1={LAMBDA1:.2f}$, $N=10000$, $a={ACTIVITY}$)   "
        rf"Corrected formula: $\tau_c^{{\rm corr}}(m) = C_0/(\lambda_1+2am)$, "
        rf"$C_0 = \tau_c^{{\rm emp}}(m{{=}}0)\cdot\lambda_1 \approx {C0:.2f}$",
        fontsize=11)
    fig.tight_layout()

    out = os.path.join("figures", "BA", "BA_zoom_E1.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved {out}")
    print(f"Theory m_c={mc_theory},  empirical m_c~{mc_emp}")


if __name__ == "__main__":
    main()
