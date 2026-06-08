"""
Network generation and loading utilities.

Generates ER, WS, and BA random graphs with matched average degree,
computes their spectral radius λ₁, and saves/loads adjacency lists.
"""

import os
import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh


def generate_networks(N, params, graph_dir, seed=42):
    """
    Generate ER, WS, and BA networks and save as adjacency lists.

    Returns a dict: {"ER": G, "WS": G, "BA": G}
    """
    os.makedirs(graph_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    graphs = {}

    for name, p in params.items():
        path = os.path.join(graph_dir, f"{name}-{N}.adjlist")
        if os.path.exists(path):
            print(f"  Loading {name} from {path}")
            G = nx.read_adjlist(path, nodetype=int)
        else:
            print(f"  Generating {name} (N={N})...")
            seed_int = int(rng.integers(0, 2**31))
            if name == "ER":
                G = nx.erdos_renyi_graph(N, p["p"], seed=seed_int)
            elif name == "WS":
                G = nx.watts_strogatz_graph(N, p["k"], p["p_rewire"], seed=seed_int)
            elif name == "BA":
                G = nx.barabasi_albert_graph(N, p["m0"], seed=seed_int)
            else:
                raise ValueError(f"Unknown network type: {name}")

            # Relabel to 0..N-1 integers
            G = nx.convert_node_labels_to_integers(G)
            nx.write_adjlist(G, path)
            print(f"  Saved to {path}")

        graphs[name] = G

    return graphs


def spectral_radius(G):
    """
    Compute the largest eigenvalue λ₁ of the adjacency matrix.
    Uses sparse ARPACK (fast for large sparse matrices).
    """
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=np.float64)
    vals = eigsh(A, k=1, which="LM", return_eigenvectors=False)
    return float(vals[0])


def network_stats(G):
    """Return a dict with basic structural statistics."""
    degrees = [d for _, d in G.degree()]
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=np.float64)
    lam1 = spectral_radius(G)
    return {
        "N": G.number_of_nodes(),
        "E": G.number_of_edges(),
        "avg_degree": float(np.mean(degrees)),
        "lambda1": lam1,
        "tau_c_static": 1.0 / lam1,
    }
