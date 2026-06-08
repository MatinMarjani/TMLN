"""
BA E4 sweep: fixed m=50, tau=0.040..0.100, to get a proper growth-rate plot.

The corrected threshold at m=50 is tau_c_corr=1.4236/(16.3635+10)~0.054,
so this range straddles the threshold cleanly.
"""

import sys, os, time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from simulation.engine import run_sis
from analysis.metrics import compute_metrics, theoretical_growth_rate, theoretical_threshold

M_FIXED  = 50
TAU_VALUES = np.round(np.arange(0.040, 0.105, 0.005), 4)   # 13 points
N_RUNS   = 100
T        = 500
T_BURN   = 250
MU       = 1.0
ACTIVITY = 0.1
N        = 10_000
LAMBDA1  = 16.3635
I0_FRAC  = 0.001   # tiny seed so surviving runs show true exponential growth

GRAPH_PATH = os.path.abspath("networks/graphs/BA-10000.adjlist")
CSV_OUT    = os.path.join(config.DATA_DIR, "BA_e4.csv")
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
        "beta": round(tau * mu, 6), "mu": mu, "activity": activity,
        "lambda1": round(lambda1, 6), "tau_c_mf": round(tau_c, 6),
        "rho_naive":       round(met["rho_naive"],       6),
        "rho_surviving":   round(met["rho_surviving"],   6),
        "extinction_prob": round(met["extinction_prob"], 4),
        "r_sim": round(r_sim, 6) if not np.isnan(r_sim) else np.nan,
        "r_th":  round(r_th,  6),
        "n_surviving_ss": round(met["n_surviving_ss"], 2),
        "n_runs": n_runs, "T": T, "T_burn": T_burn,
    }


def main():
    os.makedirs(config.DATA_DIR, exist_ok=True)

    existing_rows, done_tau = [], set()
    if os.path.exists(CSV_OUT):
        df_ex = pd.read_csv(CSV_OUT)
        existing_rows = df_ex.to_dict("records")
        done_tau = set(df_ex["tau"].round(4).tolist())
        print(f"Resuming: {len(done_tau)} tau values already done.")

    tasks = []
    for idx, tau in enumerate(TAU_VALUES):
        if round(tau, 4) in done_tau:
            continue
        seed = 42 + idx * 997
        tasks.append((GRAPH_PATH, M_FIXED, tau, MU, T, N_RUNS, I0_FRAC,
                      T_BURN, ACTIVITY, LAMBDA1, seed))

    print(f"BA E4 sweep: m={M_FIXED}, tau={TAU_VALUES[0]}..{TAU_VALUES[-1]}, workers={N_WORKERS}")
    print(f"Tasks remaining: {len(tasks)} / {len(TAU_VALUES)}")
    if not tasks:
        print("Nothing to do.")
        _plot(existing_rows)
        return

    print("Probing one point...")
    t0 = time.perf_counter()
    _worker(tasks[0])
    probe_t = time.perf_counter() - t0
    est = probe_t * len(tasks) / N_WORKERS
    print(f"  {probe_t:.1f}s/point => est. {est/60:.1f} min with {N_WORKERS} workers\n")

    rows = list(existing_rows)
    done_set = set(done_tau)

    t_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_worker, t): t for t in tasks[1:]}

        r = _worker(tasks[0])
        rows.append(r)
        done_set.add(round(r["tau"], 4))
        pd.DataFrame(rows).to_csv(CSV_OUT, index=False)

        for future in as_completed(futures):
            try:
                r = future.result()
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
            rows.append(r)
            done_set.add(round(r["tau"], 4))

            elapsed = time.perf_counter() - t_start
            n_done  = len(done_set) - len(done_tau)
            rate    = n_done / elapsed if elapsed > 0 else 1e-9
            left    = (len(tasks) - n_done) / rate / 60
            rs_str  = f"{r['r_sim']:.4f}" if isinstance(r["r_sim"], float) and not np.isnan(r["r_sim"]) else "  nan"
            print(f"  tau={r['tau']:.3f}  r_sim={rs_str}  surv={r['rho_surviving']:.4f}"
                  f"  ext={r['extinction_prob']:.2f}  [{left:.1f}min left]")
            pd.DataFrame(rows).to_csv(CSV_OUT, index=False)

    _plot(rows)


def _plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(rows).sort_values("tau")

    C0 = 0.087 * LAMBDA1
    tau_c_corr = C0 / (LAMBDA1 + 2 * ACTIVITY * M_FIXED)
    tau_c_mf   = 1.0 / (LAMBDA1 + 2 * ACTIVITY * M_FIXED)

    tau_range = np.linspace(df["tau"].min(), df["tau"].max(), 200)
    r_th_line = MU * (tau_range * (LAMBDA1 + 2 * ACTIVITY * M_FIXED) - 1)

    plt.rcParams.update({"font.size": 14, "axes.labelsize": 16,
                         "axes.titlesize": 15, "figure.dpi": 200})
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    valid = df.dropna(subset=["r_sim"])
    ax.scatter(valid["tau"], valid["r_sim"], color="steelblue", zorder=5,
               label=r"$r_{\rm sim}$ (measured)", s=40)
    ax.plot(tau_range, r_th_line, color="crimson",
            label=r"$r_{\rm th}=\mu(\tau(\lambda_1+2am)-1)$")
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.axvline(tau_c_corr, color="gray", ls="--", lw=1.5,
               label=rf"$\tau_c^{{\rm corr}}={tau_c_corr:.3f}$ ($C_0={C0:.2f}$)")

    ax.set_xlabel(r"$\tau = \beta/\mu$")
    ax.set_ylabel("Early-time growth rate $r$")
    ax.set_title(rf"BA — Growth rate at $m={M_FIXED}$")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = os.path.join("figures", "BA", f"BA_E4_growth_rate_m{M_FIXED}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved {out}")
    print(f"tau_c_corr={tau_c_corr:.4f}  tau_c_mf={tau_c_mf:.4f}")


if __name__ == "__main__":
    main()
