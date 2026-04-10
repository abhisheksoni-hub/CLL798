#!/usr/bin/env python3
"""
Avalanche Analysis for Political Opinion Dynamics Model
========================================================
Generates synthetic avalanche data (or reads NetLogo CSV export),
plots avalanche-size distributions on log-log axes, fits power laws,
and tests for self-organized criticality.

Requirements: pip install numpy matplotlib scipy
Optional:     pip install powerlaw  (for rigorous MLE fitting)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter
import os

# =============================================================================
# 1. SIMULATION (standalone Python version for batch analysis)
# =============================================================================

def build_ba_network(n, m=3):
    """Build Barabasi-Albert scale-free network. Returns adjacency list."""
    adj = {i: set() for i in range(n)}
    # Seed: complete graph on m+1 nodes
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            adj[i].add(j)
            adj[j].add(i)
    # Preferential attachment for remaining nodes
    degree = [len(adj[i]) for i in range(m + 1)] + [0] * (n - m - 1)
    for new_node in range(m + 1, n):
        targets = set()
        total_deg = sum(degree[:new_node])
        if total_deg == 0:
            targets = set(np.random.choice(new_node, size=m, replace=False))
        else:
            probs = np.array(degree[:new_node], dtype=float)
            probs /= probs.sum()
            chosen = np.random.choice(new_node, size=m, replace=False, p=probs)
            targets = set(chosen)
        for t in targets:
            adj[new_node].add(t)
            adj[t].add(new_node)
            degree[t] += 1
            degree[new_node] += 1
    return adj


def build_er_network(n, mean_k=6):
    """Build Erdos-Renyi random network. Returns adjacency list."""
    adj = {i: set() for i in range(n)}
    p_edge = mean_k / (n - 1)
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.random() < p_edge:
                adj[i].add(j)
                adj[j].add(i)
    return adj


def simulate(n=500, mean_k=6, p_noise=0.005, threshold=0.5,
             steps=2000, network="scale-free"):
    """Run opinion dynamics simulation. Returns list of avalanche sizes."""
    # Build network
    m = max(1, mean_k // 2)
    if network == "scale-free":
        adj = build_ba_network(n, m)
    else:
        adj = build_er_network(n, mean_k)

    # Initialize opinions randomly
    opinions = np.random.choice([-1, 1], size=n)
    avalanche_sizes = []
    opinion_history = []

    for t in range(steps):
        flipped = set()

        # Phase 1: Noise
        noise_mask = np.random.random(n) < p_noise
        for i in np.where(noise_mask)[0]:
            opinions[i] *= -1
            flipped.add(int(i))

        # Phase 2: Cascade
        active = set(flipped)
        while active:
            next_active = set()
            candidates = set()
            for i in active:
                for j in adj[i]:
                    if j not in flipped:
                        candidates.add(j)

            for j in candidates:
                neighbors = adj[j]
                if len(neighbors) == 0:
                    continue
                disagreeing = sum(1 for nb in neighbors if opinions[nb] != opinions[j])
                frac = disagreeing / len(neighbors)
                if frac > threshold:
                    opinions[j] *= -1
                    flipped.add(j)
                    next_active.add(j)

            active = next_active

        avalanche_sizes.append(len(flipped))
        opinion_history.append(np.mean(opinions > 0))

    return avalanche_sizes, opinion_history


# =============================================================================
# 2. ANALYSIS FUNCTIONS
# =============================================================================

def compute_ccdf(sizes):
    """Compute complementary cumulative distribution function."""
    sizes = np.array([s for s in sizes if s > 0])
    sorted_s = np.sort(sizes)
    ccdf = 1.0 - np.arange(1, len(sorted_s) + 1) / len(sorted_s)
    return sorted_s, ccdf


def log_binned_pdf(sizes, num_bins=20):
    """Compute logarithmically binned probability density."""
    sizes = np.array([s for s in sizes if s > 0])
    if len(sizes) == 0:
        return np.array([]), np.array([])
    bins = np.logspace(np.log10(1), np.log10(max(sizes) + 1), num_bins + 1)
    counts, edges = np.histogram(sizes, bins=bins, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])  # geometric mean
    mask = counts > 0
    return centers[mask], counts[mask]


def fit_power_law_ols(sizes, s_min=2):
    """Fit power law via OLS on log-log CCDF. Returns exponent and R^2."""
    s_arr, ccdf = compute_ccdf(sizes)
    mask = (s_arr >= s_min) & (ccdf > 0)
    if mask.sum() < 5:
        return None, None
    log_s = np.log10(s_arr[mask])
    log_c = np.log10(ccdf[mask])
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_s, log_c)
    # CCDF exponent = -(tau - 1), so tau = 1 - slope
    tau = 1 - slope
    return tau, r_value**2


def fit_power_law_mle(sizes, s_min=2):
    """Estimate power-law exponent via maximum likelihood (discrete)."""
    data = np.array([s for s in sizes if s >= s_min])
    if len(data) < 10:
        return None
    # Discrete MLE: tau = 1 + n / sum(ln(s / (s_min - 0.5)))
    n = len(data)
    tau = 1 + n / np.sum(np.log(data / (s_min - 0.5)))
    return tau


# =============================================================================
# 3. PLOTTING
# =============================================================================

def plot_avalanche_analysis(ba_sizes, er_sizes, opinion_ba, opinion_er):
    """Generate publication-quality analysis figures."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # --- Panel A: CCDF log-log plot ---
    ax = axes[0, 0]
    s_ba, ccdf_ba = compute_ccdf(ba_sizes)
    s_er, ccdf_er = compute_ccdf(er_sizes)
    ax.loglog(s_ba, ccdf_ba, 'o', markersize=3, alpha=0.6, color='#2166ac',
              label='Scale-free (BA)')
    ax.loglog(s_er, ccdf_er, 's', markersize=3, alpha=0.6, color='#b2182b',
              label='Random (ER)')
    # Fit line for BA
    tau_ba, r2_ba = fit_power_law_ols(ba_sizes)
    if tau_ba is not None:
        x_fit = np.logspace(0, np.log10(max(s_ba)), 50)
        # CCDF ~ s^{-(tau-1)}
        y_fit = x_fit**(-(tau_ba - 1))
        y_fit *= ccdf_ba[0] / y_fit[0]  # normalize
        ax.loglog(x_fit, y_fit, '--', color='#2166ac', alpha=0.8,
                  label=f'Fit: τ={tau_ba:.2f} (R²={r2_ba:.3f})')
    ax.set_xlabel('Avalanche size s')
    ax.set_ylabel('P(S ≥ s)')
    ax.set_title('(A) Avalanche Size CCDF')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel B: Log-binned PDF ---
    ax = axes[0, 1]
    c_ba, p_ba = log_binned_pdf(ba_sizes)
    c_er, p_er = log_binned_pdf(er_sizes)
    if len(c_ba) > 0:
        ax.loglog(c_ba, p_ba, 'o-', markersize=5, color='#2166ac', label='Scale-free (BA)')
    if len(c_er) > 0:
        ax.loglog(c_er, p_er, 's-', markersize=5, color='#b2182b', label='Random (ER)')
    ax.set_xlabel('Avalanche size s')
    ax.set_ylabel('P(s)')
    ax.set_title('(B) Log-Binned PDF')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel C: Opinion fraction time series ---
    ax = axes[1, 0]
    ax.plot(opinion_ba, color='#2166ac', alpha=0.7, linewidth=0.8, label='Scale-free (BA)')
    ax.plot(opinion_er, color='#b2182b', alpha=0.7, linewidth=0.8, label='Random (ER)')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Fraction with opinion +1')
    ax.set_title('(C) Opinion Fraction Time Series')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel D: Avalanche size time series ---
    ax = axes[1, 1]
    ax.plot(ba_sizes, color='#2166ac', alpha=0.6, linewidth=0.5, label='Scale-free (BA)')
    ax.plot(er_sizes, color='#b2182b', alpha=0.6, linewidth=0.5, label='Random (ER)')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Avalanche size')
    ax.set_title('(D) Avalanche Size Over Time')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('avalanche_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: avalanche_analysis.png")


# =============================================================================
# 4. MAIN
# =============================================================================

def main():
    np.random.seed(42)

    print("=" * 60)
    print("Political Opinion Dynamics — Avalanche Analysis")
    print("=" * 60)

    # Parameters
    N = 500
    MEAN_K = 6
    P_NOISE = 0.005
    THRESHOLD = 0.5
    STEPS = 2000

    # --- Run simulations ---
    print(f"\nSimulating BA scale-free network (N={N}, steps={STEPS})...")
    ba_sizes, opinion_ba = simulate(N, MEAN_K, P_NOISE, THRESHOLD, STEPS, "scale-free")

    print(f"Simulating ER random network (N={N}, steps={STEPS})...")
    er_sizes, opinion_er = simulate(N, MEAN_K, P_NOISE, THRESHOLD, STEPS, "random")

    # --- Analysis ---
    ba_nonzero = [s for s in ba_sizes if s > 0]
    er_nonzero = [s for s in er_sizes if s > 0]

    print(f"\n--- Scale-Free (BA) Network ---")
    print(f"  Non-zero avalanches: {len(ba_nonzero)} / {STEPS}")
    print(f"  Mean avalanche size: {np.mean(ba_nonzero):.2f}" if ba_nonzero else "  No avalanches")
    print(f"  Max avalanche size:  {max(ba_nonzero)}" if ba_nonzero else "")
    tau_ba_ols, r2_ba = fit_power_law_ols(ba_sizes)
    tau_ba_mle = fit_power_law_mle(ba_sizes)
    if tau_ba_ols:
        print(f"  Power-law exponent (OLS):  τ = {tau_ba_ols:.3f}  (R² = {r2_ba:.4f})")
    if tau_ba_mle:
        print(f"  Power-law exponent (MLE):  τ = {tau_ba_mle:.3f}")

    print(f"\n--- Random (ER) Network ---")
    print(f"  Non-zero avalanches: {len(er_nonzero)} / {STEPS}")
    print(f"  Mean avalanche size: {np.mean(er_nonzero):.2f}" if er_nonzero else "  No avalanches")
    print(f"  Max avalanche size:  {max(er_nonzero)}" if er_nonzero else "")
    tau_er_ols, r2_er = fit_power_law_ols(er_sizes)
    tau_er_mle = fit_power_law_mle(er_sizes)
    if tau_er_ols:
        print(f"  Power-law exponent (OLS):  τ = {tau_er_ols:.3f}  (R² = {r2_er:.4f})")
    if tau_er_mle:
        print(f"  Power-law exponent (MLE):  τ = {tau_er_mle:.3f}")

    # --- Interpretation ---
    print("\n--- SOC Interpretation ---")
    if tau_ba_ols and r2_ba and r2_ba > 0.9:
        print("  BA network: Strong evidence for power-law scaling (SOC-like behavior).")
    elif tau_ba_ols and r2_ba and r2_ba > 0.8:
        print("  BA network: Moderate evidence for power-law scaling.")
    else:
        print("  BA network: Weak/no power-law scaling detected. Try adjusting parameters.")

    if tau_er_ols and r2_er and r2_er > 0.9:
        print("  ER network: Power-law detected (unexpected for homogeneous networks).")
    else:
        print("  ER network: No power-law scaling — consistent with subcritical dynamics.")

    # --- Generate plots ---
    print("\nGenerating plots...")
    plot_avalanche_analysis(ba_sizes, er_sizes, opinion_ba, opinion_er)

    # --- Save data to CSV ---
    with open("avalanche_data_python.csv", "w") as f:
        f.write("step,ba_avalanche_size,er_avalanche_size,ba_opinion_frac,er_opinion_frac\n")
        for i in range(STEPS):
            f.write(f"{i},{ba_sizes[i]},{er_sizes[i]},{opinion_ba[i]:.4f},{opinion_er[i]:.4f}\n")
    print("Saved: avalanche_data_python.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
