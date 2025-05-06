import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor

# === Parameters ===
adjlist = 'BA-10000'
static_graph = nx.read_adjlist(f"../{adjlist}.txt", nodetype=int)
N = static_graph.number_of_nodes()
static_edges_set = set(static_graph.edges())

m_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
activity = 0.01
activity_rates = {node: activity for node in static_graph.nodes}

timesteps = 150
num_runs = 100

beta = 0.08
mu = 0.1
initial_infected = set(random.sample(range(N), int(0.05 * N)))

csv_file = "sir_summary_results.csv"

# Precompute candidates for temporal edges
candidate_neighbors = {i: [j for j in range(N) if j != i and (i, j) not in static_edges_set and (j, i) not in static_edges_set] for i in range(N)}

# === Run Simulation Function ===
def run_sir_once(args):
    m = args[0]
    states = {i: 'I' if i in initial_infected else 'S' for i in range(N)}
    counts = {'S': [], 'I': [], 'R': []}
    extinction_time = timesteps

    for step in range(timesteps):
        G_temporal = nx.Graph()
        G_temporal.add_nodes_from(range(N))

        for i in range(N):
            if random.random() < activity_rates[i]:
                targets = random.sample(candidate_neighbors[i], min(m, len(candidate_neighbors[i])))
                G_temporal.add_edges_from((i, j) for j in targets)

        infected_set = {n for n, state in states.items() if state == 'I'}
        new_states = states.copy()

        for node in range(N):
            if states[node] == 'I':
                if random.random() < mu:
                    new_states[node] = 'R'
            elif states[node] == 'S':
                infected_neighbors = 0

                # static neighbors
                for neighbor in static_graph.neighbors(node):
                    if neighbor in infected_set:
                        infected_neighbors += 1

                # temporal neighbors
                for neighbor in G_temporal.neighbors(node):
                    if neighbor in infected_set:
                        infected_neighbors += 1

                if infected_neighbors > 0:
                    p_infection = 1 - (1 - beta) ** infected_neighbors
                    if random.random() < p_infection:
                        new_states[node] = 'I'

        states = new_states
        current_I = sum(1 for s in states.values() if s == 'I')
        counts['S'].append(sum(1 for s in states.values() if s == 'S'))
        counts['I'].append(current_I)
        counts['R'].append(sum(1 for s in states.values() if s == 'R'))

        if current_I == 0 and extinction_time == timesteps:
            extinction_time = step

    return counts, extinction_time

# === Main Loop ===
if __name__ == "__main__":
    for m in m_values:
        print(f"Running {num_runs} SIR simulations for m = {m}...")
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(run_sir_once, [(m,)] * num_runs))

        all_S = [r[0]['S'] for r in results]
        all_I = [r[0]['I'] for r in results]
        all_R = [r[0]['R'] for r in results]
        extinction_times = [r[1] for r in results]
        peak_times = [np.argmax(r[0]['I']) for r in results]
        peak_values = [max(r[0]['I']) for r in results]

        all_S = np.array(all_S) / N
        all_I = np.array(all_I) / N
        all_R = np.array(all_R) / N

        mean_S = np.mean(all_S, axis=0)
        mean_I = np.mean(all_I, axis=0)
        mean_R = np.mean(all_R, axis=0)

        std_S = np.std(all_S, axis=0)
        std_I = np.std(all_I, axis=0)
        std_R = np.std(all_R, axis=0)

        avg_extinction_time = np.mean(extinction_times)
        std_extinction_time = np.std(extinction_times)
        avg_peak_time = np.mean(peak_times)
        std_peak_time = np.std(peak_times)
        avg_peak_val = np.mean(peak_values)
        std_peak_val = np.std(peak_values)

        t = np.arange(timesteps)
        plt.figure(figsize=(10, 6))
        summary_text = (
            f"$\\beta$ = {beta}, $\\mu$ = {mu}, m = {m}\n"
            f"Extinction: {avg_extinction_time:.2f} ± {std_extinction_time:.2f} steps\n"
            f"Peak Time: {avg_peak_time:.2f} ± {std_peak_time:.2f}\n"
            f"Peak Size: {avg_peak_val:.2f} ± {std_peak_val:.2f}\n"
            f"Max Recovered: {mean_R.max():.2f}"
        )
        plt.gcf().text(0.02, 0.75, summary_text, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
        plt.plot(t, mean_S, label='Susceptible', color='skyblue', linewidth=2)
        plt.fill_between(t, mean_S - std_S, mean_S + std_S, color='skyblue', alpha=0.3)
        plt.plot(t, mean_I, label='Infected', color='red', linewidth=2)
        plt.fill_between(t, mean_I - std_I, mean_I + std_I, color='red', alpha=0.3)
        plt.plot(t, mean_R, label='Recovered', color='gray', linewidth=2)
        plt.fill_between(t, mean_R - std_R, mean_R + std_R, color='gray', alpha=0.3)
        plt.title(f"Average SIR (m={m})")
        plt.xlabel("Time Step")
        plt.ylabel("Fraction of Population")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"sir_average_curve_m={m}.png", dpi=300)
        plt.close()
        print(f"Saved SIR plot as sir_average_curve_m={m}.png")

        summary_data = {
            'Network': adjlist,
            'm': [m],
            'beta': [beta],
            'mu': [mu],
            'extinction_time': [f"{avg_extinction_time:.2f} +/- {std_extinction_time:.2f}"],
            'peak_time': [f"{avg_peak_time:.2f} +/- {std_peak_time:.2f}"],
            'peak_size': [f"{avg_peak_val:.2f} +/- {std_peak_val:.2f}"],
            'max_R_mean': [mean_R.max()],
        }
        df = pd.DataFrame(summary_data)
        if os.path.exists(csv_file):
            existing_df = pd.read_csv(csv_file)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
        else:
            combined_df = df
        combined_df.to_csv(csv_file, index=False)
        print(f"Saved summary to {csv_file}")

