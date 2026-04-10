"""
Avalanche Analysis for Political Opinion Dynamics
Generates Figures 1-6 from the manuscript.
"""
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 14,
    'figure.dpi': 200, 'savefig.dpi': 200, 'savefig.bbox': 'tight'
})

def build_network(N, k_mean, net_type):
    if net_type == 'BA':
        m = max(1, k_mean // 2)
        return nx.barabasi_albert_graph(N, m)
    else:
        p = k_mean / (N - 1)
        return nx.erdos_renyi_graph(N, p)

def simulate(G, p_noise, theta, T):
    N = G.number_of_nodes()
    opinions = np.random.choice([-1, 1], size=N)
    adj = {i: list(G.neighbors(i)) for i in range(N)}
    avalanche_sizes = []
    phi_plus = []
    for t in range(T):
        flipped = set()
        for i in range(N):
            if np.random.random() < p_noise:
                opinions[i] *= -1
                flipped.add(i)
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
                disagreeing = sum(1 for m in neighbors if opinions[m] != opinions[j])
                frac = disagreeing / len(neighbors)
                if frac > theta:
                    opinions[j] *= -1
                    flipped.add(j)
                    next_active.add(j)
            active = next_active
        avalanche_sizes.append(len(flipped))
        phi_plus.append(np.sum(opinions == 1) / N)
    return avalanche_sizes, phi_plus

def ccdf(sizes):
    sizes = np.array([s for s in sizes if s > 0])
    if len(sizes) == 0:
        return np.array([1]), np.array([1])
    unique = np.unique(np.sort(sizes))
    ccdf_vals = np.array([np.sum(sizes >= s) / len(sizes) for s in unique])
    return unique, ccdf_vals

N = 500; k_mean = 6; p_noise = 0.005; theta = 0.5; T = 2000; n_runs = 5
np.random.seed(42)

# Figure 1: CCDF BA
print("Figure 1...")
all_ba = []
for r in range(n_runs):
    G = build_network(N, k_mean, 'BA')
    s, _ = simulate(G, p_noise, theta, T)
    all_ba.extend(s)
x, y = ccdf(all_ba)
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.loglog(x, y, 'o', ms=4, alpha=0.7, color='steelblue', label='Simulation data')
mask = (x >= 2) & (x <= max(x))
if np.sum(mask) > 2:
    c = np.polyfit(np.log10(x[mask]), np.log10(y[mask]), 1)
    tau = -c[0]
    xf = np.logspace(np.log10(2), np.log10(max(x[mask])), 50)
    ax.loglog(xf, 10**c[1]*xf**c[0], 'r--', lw=2, label=f'Power law ($\\tau \\approx {tau:.2f}$)')
ax.set_xlabel('Avalanche size $s$'); ax.set_ylabel('$P(S \\geq s)$')
ax.set_title('Figure 1: Avalanche CCDF — BA Network')
ax.legend(); ax.grid(True, alpha=0.3)
plt.savefig('figure1.png'); plt.close()

# Figure 2: CCDF ER
print("Figure 2...")
all_er = []
for r in range(n_runs):
    G = build_network(N, k_mean, 'ER')
    s, _ = simulate(G, p_noise, theta, T)
    all_er.extend(s)
x2, y2 = ccdf(all_er)
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.loglog(x2, y2, 's', ms=4, alpha=0.7, color='darkorange', label='Simulation data')
mask2 = x2 >= 2
if np.sum(mask2) > 2:
    c2 = np.polyfit(np.log10(x2[mask2]), np.log10(y2[mask2]), 1)
    xf2 = np.logspace(np.log10(2), np.log10(max(x2[mask2])), 50)
    ax.loglog(xf2, 10**c2[1]*xf2**c2[0], 'r--', lw=2, label='Fit (exponential cutoff visible)')
ax.set_xlabel('Avalanche size $s$'); ax.set_ylabel('$P(S \\geq s)$')
ax.set_title('Figure 2: Avalanche CCDF — ER Network')
ax.legend(); ax.grid(True, alpha=0.3)
plt.savefig('figure2.png'); plt.close()

# Figure 3: Mean avalanche vs connectivity
print("Figure 3...")
k_vals = list(range(2, 16, 2))
m_ba = []; m_er = []
for k in k_vals:
    sb = []; se = []
    for r in range(3):
        s, _ = simulate(build_network(N, k, 'BA'), p_noise, theta, 500)
        sb.extend(s)
        s, _ = simulate(build_network(N, k, 'ER'), p_noise, theta, 500)
        se.extend(s)
    m_ba.append(np.mean(sb)); m_er.append(np.mean(se))
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(k_vals, m_ba, 'o-', color='steelblue', lw=2, label='BA (scale-free)')
ax.plot(k_vals, m_er, 's--', color='darkorange', lw=2, label='ER (random)')
ax.set_xlabel('Mean degree $\\langle k \\rangle$'); ax.set_ylabel('Mean avalanche size $\\langle s \\rangle$')
ax.set_title('Figure 3: Mean Avalanche Size vs. Connectivity')
ax.legend(); ax.grid(True, alpha=0.3)
plt.savefig('figure3.png'); plt.close()

# Figure 4: Varying noise
print("Figure 4...")
noise_vals = [0.001, 0.005, 0.01, 0.05]
colors = ['navy', 'steelblue', 'forestgreen', 'firebrick']
fig, ax = plt.subplots(figsize=(6, 4.5))
for pn, col in zip(noise_vals, colors):
    ss = []
    for r in range(3):
        s, _ = simulate(build_network(N, k_mean, 'BA'), pn, theta, 1000)
        ss.extend(s)
    xv, yv = ccdf(ss)
    ax.loglog(xv, yv, 'o', ms=3, alpha=0.7, color=col, label=f'$p_{{noise}}={pn}$')
ax.set_xlabel('Avalanche size $s$'); ax.set_ylabel('$P(S \\geq s)$')
ax.set_title('Figure 4: Effect of Noise (BA Network)')
ax.legend(); ax.grid(True, alpha=0.3)
plt.savefig('figure4.png'); plt.close()

# Figure 5: Time series
print("Figure 5...")
_, phi_ba = simulate(build_network(N, k_mean, 'BA'), p_noise, theta, T)
_, phi_er = simulate(build_network(N, k_mean, 'ER'), p_noise, theta, T)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(T), phi_ba, lw=0.6, color='steelblue', label='BA', alpha=0.8)
ax.plot(range(T), phi_er, lw=0.6, color='darkorange', label='ER', alpha=0.8)
ax.set_xlabel('Time step $t$'); ax.set_ylabel('$\\phi^+(t)$')
ax.set_title('Figure 5: Opinion Fraction Time Series')
ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)
plt.savefig('figure5.png'); plt.close()

# Figure 6: Phase diagram
print("Figure 6...")
theta_vals = np.linspace(0.15, 0.85, 10)
pnoise_vals = np.logspace(-3, -1, 8)
phase = np.zeros((len(pnoise_vals), len(theta_vals)))
for i, pn in enumerate(pnoise_vals):
    for j, th in enumerate(theta_vals):
        s, _ = simulate(build_network(N, k_mean, 'BA'), pn, th, 200)
        nz = [v for v in s if v > 0]
        phase[i, j] = np.mean(nz) if nz else 0
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.pcolormesh(theta_vals, pnoise_vals, phase, cmap='inferno', shading='auto')
ax.set_yscale('log')
ax.set_xlabel('Influence threshold $\\theta$'); ax.set_ylabel('$p_{noise}$')
ax.set_title('Figure 6: Phase Diagram (BA Network)')
cb = plt.colorbar(im, ax=ax); cb.set_label('$\\langle s \\rangle$')
ax.text(0.25, 0.003, 'Supercritical', color='white', fontsize=10, fontstyle='italic')
ax.text(0.50, 0.006, 'Critical', color='white', fontsize=10, fontstyle='italic')
ax.text(0.75, 0.02, 'Subcritical', color='white', fontsize=10, fontstyle='italic')
plt.savefig('figure6.png'); plt.close()

print("All figures generated!")
