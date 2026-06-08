"""
Main sweep script: run the (m, tau) grid for one network family and save results.

Usage (from SIS_Revision/ with venv active):
    python scripts/run_sweep.py --network ER
    python scripts/run_sweep.py --network WS
    python scripts/run_sweep.py --network BA

Options:
    --workers N   parallel worker processes (default: half of CPU count)
    --test        quick smoke-test with a small grid

Output:
    data/<NETWORK>_results.csv  — one row per (m, tau) point.

CSV columns:
    network, m, tau, beta, mu, activity,
    rho_naive, rho_surviving, extinction_prob, r_sim, r_th, n_surviving_ss,
    lambda1, tau_c_mf, n_runs, T, T_burn
"""

import sys
import os
import argparse
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from networks.generate import generate_networks, network_stats
from simulation.engine import run_sis
from analysis.metrics import compute_metrics, theoretical_growth_rate, theoretical_threshold

import networkx as nx


# ── Worker function ───────────────────────────────────────────────────────────
# Runs in a separate process; receives everything it needs as plain arguments.

def _worker(args):
    (graph_path, N, a, m, tau, mu, T, n_runs, I0_frac, T_burn,
     early_frac, early_min_pts, lambda1, seed) = args

    G = nx.read_adjlist(graph_path, nodetype=int)
    beta = tau * mu

    I_series = run_sis(
        G_static=G, a=a, m=m, beta=beta, mu=mu,
        T=T, n_runs=n_runs, I0_frac=I0_frac, seed=seed,
    )
    metrics = compute_metrics(
        I_series, N=N, T_burn=T_burn,
        early_growth_frac=early_frac,
        early_growth_min_points=early_min_pts,
    )
    r_th  = theoretical_growth_rate(mu, tau, lambda1, a, m)
    tau_c = theoretical_threshold(lambda1, a, m)

    r_sim = metrics["r_sim"]
    return {
        "m":              m,
        "tau":            round(float(tau), 6),
        "beta":           round(float(beta), 6),
        "mu":             mu,
        "activity":       a,
        "lambda1":        round(lambda1, 6),
        "tau_c_mf":       round(tau_c, 6),
        "rho_naive":      round(metrics["rho_naive"], 6),
        "rho_surviving":  round(metrics["rho_surviving"], 6),
        "extinction_prob":round(metrics["extinction_prob"], 4),
        "r_sim":          round(r_sim, 6) if not np.isnan(r_sim) else np.nan,
        "r_th":           round(r_th, 6),
        "n_surviving_ss": round(metrics["n_surviving_ss"], 2),
        "n_runs":         n_runs,
        "T":              T,
        "T_burn":         T_burn,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--network", choices=["ER", "WS", "BA"], required=True)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2),
                   help="Parallel worker processes (default: CPU_count/2)")
    p.add_argument("--test", action="store_true",
                   help="Smoke-test: small grid, few runs (~30s)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    net = args.network

    if args.test:
        m_values   = [0, 5, 10]
        tau_values = [0.10, 0.15, 0.20, 0.25, 0.30]
        n_runs, timesteps, t_burn = 20, 200, 100
        print("  [TEST MODE] 3 m-values x 5 tau-values, 20 runs")
    else:
        m_values   = config.M_VALUES
        tau_values = config.TAU_VALUES
        n_runs     = config.N_RUNS
        timesteps  = config.TIMESTEPS
        t_burn     = config.T_BURN

    # ── Network ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" Network: {net}   N={config.N}   workers={args.workers}")
    print(f"{'='*60}")

    graphs = generate_networks(
        config.N, {net: config.NETWORK_PARAMS[net]},
        config.GRAPH_DIR, seed=config.GRAPH_SEED,
    )
    stats = network_stats(graphs[net])
    lambda1 = stats["lambda1"]
    graph_path = os.path.abspath(
        os.path.join(config.GRAPH_DIR, f"{net}-{config.N}.adjlist")
    )
    print(f"  avg_degree = {stats['avg_degree']:.3f}")
    print(f"  lambda1    = {lambda1:.4f}   tau_c(m=0) = {stats['tau_c_static']:.4f}")

    # ── Build task list (supports resuming) ───────────────────────────────────
    os.makedirs(config.DATA_DIR, exist_ok=True)
    csv_path = os.path.join(config.DATA_DIR, f"{net}_results.csv")

    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        done = set(zip(
            existing_df["m"].astype(int).tolist(),
            existing_df["tau"].round(6).tolist(),
        ))
        rows = existing_df.to_dict("records")
        print(f"  Resuming: {len(done)} points already done.")
    else:
        done, rows = set(), []

    tasks = []
    for idx, m in enumerate(m_values):
        for jdx, tau in enumerate(tau_values):
            tau_r = round(float(tau), 6)
            if (int(m), tau_r) in done:
                continue
            seed = args.seed + (idx * len(tau_values) + jdx) * 997
            tasks.append((
                graph_path, config.N, config.ACTIVITY,
                int(m), tau_r, config.MU,
                timesteps, n_runs, config.I0_FRAC, t_burn,
                config.EARLY_GROWTH_FRAC, config.EARLY_GROWTH_MIN_POINTS,
                lambda1, seed,
            ))

    total = len(m_values) * len(tau_values)
    print(f"  Tasks to run: {len(tasks)} / {total}")
    if not tasks:
        print("  Nothing to do.")
        return

    # ── Quick timing probe ────────────────────────────────────────────────────
    print(f"  Probing one point for timing estimate...")
    t0 = time.perf_counter()
    probe_row = _worker(tasks[0])
    t_point = time.perf_counter() - t0
    probe_row["network"] = net
    rows.append(probe_row)
    done.add((probe_row["m"], probe_row["tau"]))
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    remaining_tasks = tasks[1:]
    t_est = t_point * len(remaining_tasks) / args.workers
    print(f"  {t_point:.1f}s/point  ->  est. {t_est/60:.0f} min with {args.workers} workers")

    # ── Parallel sweep ────────────────────────────────────────────────────────
    completed = len(done)
    t_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_worker, task): task for task in remaining_tasks}

        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            row["network"] = net
            rows.append(row)
            completed += 1

            elapsed = time.perf_counter() - t_start
            n_done_this_session = completed - (total - len(tasks))
            rate = n_done_this_session / elapsed if elapsed > 0 else 1e-9
            left = (len(tasks) - n_done_this_session) / rate / 60

            r_sim = row["r_sim"]
            r_sim_str = f"{r_sim:.4f}" if isinstance(r_sim, float) and not np.isnan(r_sim) else "  nan "
            print(
                f"  [{completed:4d}/{total}]"
                f"  m={row['m']:2d}  tau={row['tau']:.3f}"
                f"  surv={row['rho_surviving']:.4f}"
                f"  ext={row['extinction_prob']:.2f}"
                f"  r_sim={r_sim_str}  r_th={row['r_th']:.4f}"
                f"  [{left:.0f}min left]"
            )

            pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"\nDone. Saved to {csv_path}")


if __name__ == "__main__":
    main()
