import os
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Ensure output directory exists
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# GRAPH TOPOLOGY SEED ENGINES
# =====================================================================
def get_torus_distances(L):
    x, y = np.meshgrid(np.arange(L), np.arange(L))
    pos = np.vstack([x.ravel(), y.ravel()]).T
    N = L * L
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                dx = min(abs(pos[i,0] - pos[j,0]), L - abs(pos[i,0] - pos[j,0]))
                dy = min(abs(pos[i,1] - pos[j,1]), L - abs(pos[i,1] - pos[j,1]))
                D[i,j] = np.sqrt(dx**2 + dy**2)
    return D

def generate_erdos_renyi_base(N, p=0.4):
    rng = np.random.default_rng(seed=42)
    adj = rng.random((N, N)) < p
    adj = np.triu(adj, 1) + np.triu(adj, 1).T
    D = np.where(adj, 1.0, 2.5)
    np.fill_diagonal(D, 0.0)
    return D

def generate_watts_strogatz_base(N, k=4, p=0.2):
    rng = np.random.default_rng(seed=42)
    D = np.zeros((N, N))
    for i in range(N):
        for step in range(1, k // 2 + 1):
            D[i, (i + step) % N] = 1.0
            D[i, (i - step) % N] = 1.0
    for i in range(N):
        for j in range(i + 1, N):
            if D[i, j] == 1.0 and rng.random() < p:
                D[i, j] = 2.5
                new_target = rng.integers(0, N)
                if new_target != i:
                    D[i, new_target] = 1.0
    D = np.maximum(D, D.T)
    D[D == 0.0] = 3.0
    np.fill_diagonal(D, 0.0)
    return D

def generate_barabasi_albert_base(N, m=3):
    rng = np.random.default_rng(seed=42)
    D = np.full((N, N), 3.0)
    np.fill_diagonal(D, 0.0)
    for i in range(m):
        for j in range(i + 1, m):
            D[i, j] = D[j, i] = 1.0
    for i in range(m, N):
        degrees = np.sum(D[:i, :i] == 1.0, axis=1)
        prob = degrees / np.sum(degrees)
        targets = rng.choice(np.arange(i), size=m, replace=False, p=prob)
        for t in targets:
            D[i, t] = D[t, i] = 1.0
    return D

# =====================================================================
# THERMODYNAMIC PARTITION FUNCTION CORE ENGINES
# =====================================================================
def compute_thermodynamic_metrics(eigenvals, beta=0.1):
    active_modes = eigenvals[1:]
    exp_weights = np.exp(-beta * active_modes)
    Z = np.sum(exp_weights)
    avg_lambda = np.sum(active_modes * exp_weights) / Z
    avg_lambda_sq = np.sum((active_modes**2) * exp_weights) / Z
    C_v = (beta**2) * (avg_lambda_sq - (avg_lambda**2))
    return avg_lambda, C_v

def compute_peak_spectral_dimension(eigenvals, t_space=np.logspace(-2, 1, 100)):
    d_s_max = 0.0
    for t in t_space:
        exp_terms = np.exp(-t * eigenvals)
        P_t = np.sum(exp_terms)
        P_prime_t = np.sum(eigenvals * exp_terms)
        d_s_t = 2.0 * t * (P_prime_t / P_t)
        if d_s_t > d_s_max:
            d_s_max = d_s_t
    return d_s_max

# =====================================================================
# PIPELINE GENERATORS
# =====================================================================
def make_figure_1_cosmic_timeline(L=8, time_steps=100):
    D = get_torus_distances(L)
    mask = D > 0
    a_timeline, alpha_timeline, w_timeline = [], [], []
    a, dt, a_scale = 0.01, 0.005, 1.05
    
    for _ in range(time_steps):
        alpha = 4.0 * (1.0 - np.exp(-1.5 * a))
        I_base = np.zeros_like(D)
        I_base[mask] = 1.0 / (D[mask] ** alpha) if alpha > 0 else 1.0
        rho_base, _ = compute_thermodynamic_metrics(np.sort(la.eigvalsh(np.diag(np.sum(I_base, axis=1)) - I_base)))
        
        I_scaled = np.zeros_like(D)
        I_scaled[mask] = 1.0 / ((D[mask] * a_scale) ** alpha) if alpha > 0 else 1.0
        rho_scaled, _ = compute_thermodynamic_metrics(np.sort(la.eigvalsh(np.diag(np.sum(I_scaled, axis=1)) - I_scaled)))
        
        w_eff = (-(np.log(rho_scaled / rho_base) / np.log(a_scale)) / 3.0) - 1.0
        a_timeline.append(a)
        alpha_timeline.append(alpha)
        w_timeline.append(w_eff)
        a += a * np.sqrt(rho_base) * dt

    fig, ax1 = plt.subplots(figsize=(7.5, 5), dpi=300)
    color = 'purple'
    ax1.set_xlabel('Simulated Cosmic Expansion Time ($t$)', fontsize=10)
    ax1.set_ylabel(r'Effective Equation of State $w_{\text{eff}}(t)$', color=color, fontsize=10)
    ax1.plot(np.arange(time_steps), w_timeline, color=color, linewidth=2.5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(-1.0, color='red', linestyle='--', alpha=0.5)
    ax1.axhline(1/3, color='blue', linestyle=':', alpha=0.5)
    ax1.axhline(0.0, color='green', linestyle='-.', alpha=0.5)

    ax2 = ax1.twinx()  
    color = 'darkorange'
    ax2.set_ylabel(r'Substrate Decay Index $\alpha(t)$', color=color, fontsize=10)
    ax2.plot(np.arange(time_steps), alpha_timeline, color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title("Field-Free Cosmic Timeline: The Topological Graceful Exit", fontsize=11, fontweight="bold", pad=12)
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "dynamic_cosmic_timeline_evolution.png"), bbox_inches="tight")
    plt.close()

def make_figure_2_thermodynamic_stability(L=8, steps=100):
    D = get_torus_distances(L)
    mask = D > 0
    a_space = np.linspace(0.28, 1.0, steps)
    rho_series, Cv_series = [], []
    
    for a in a_space:
        alpha = max(0.005, 4.0*(1.0-np.exp(-1.5*a)) - 1.8*np.exp(-((a-0.4)/0.05)**2))
        I = np.zeros_like(D)
        I[mask] = 1.0 / (D[mask] ** alpha)
        rho, Cv = compute_thermodynamic_metrics(np.sort(la.eigvalsh(np.diag(np.sum(I, axis=1)) - I)))
        rho_series.append(rho)
        Cv_series.append(Cv)

    fig, ax1 = plt.subplots(figsize=(7.5, 5), dpi=300)
    color = 'tab:blue'
    ax1.set_xlabel('Cosmic Scale Factor ($a$)', fontsize=10)
    ax1.set_ylabel(r'Derived Matrix Energy Density $\rho_{\text{eff}}(a)$', color=color, fontsize=10)
    ax1.plot(a_space, rho_series, color=color, linewidth=2.5)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel(r'Specific Heat Capacity $C_v(a)$', color=color, fontsize=10)
    ax2.plot(a_space, Cv_series, color=color, linestyle='-.', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axhline(0.0, color='black', linestyle='--', alpha=0.5)
    plt.title("Thermodynamic Stability Envelope of the Late-Time Rupture", fontsize=11, fontweight="bold", pad=12)
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rupture_thermodynamic_stability.png"), bbox_inches="tight")
    plt.close()

def make_figure_3_3d_rg_manifold(N=100, steps=60):
    alpha_space = np.linspace(1.5, 0.005, steps)
    topologies = ["Erdos-Renyi", "Watts-Strogatz", "Barabasi-Albert"]
    a_scale = 1.05
    results = {top: {"x_lambda": [], "y_weff": [], "z_ds": []} for top in topologies}
    matrix_factory = {"Erdos-Renyi": generate_erdos_renyi_base(N), "Watts-Strogatz": generate_watts_strogatz_base(N), "Barabasi-Albert": generate_barabasi_albert_base(N, m=3)}
    
    for alpha in alpha_space:
        for top in topologies:
            D = matrix_factory[top]
            mask = D > 0
            I = np.zeros_like(D); I[mask] = 1.0 / (D[mask] ** alpha)
            L_base = np.diag(np.sum(I, axis=1)) - I; eigenvals = np.sort(la.eigvalsh(L_base))
            
            I_scaled = np.zeros_like(D); I_scaled[mask] = 1.0 / ((D[mask] * a_scale) ** alpha)
            rho_base, _ = compute_thermodynamic_metrics(eigenvals)
            rho_scaled, _ = compute_thermodynamic_metrics(np.sort(la.eigvalsh(np.diag(np.sum(I_scaled, axis=1)) - I_scaled)))
            
            w_eff = (-(np.log(rho_scaled / rho_base) / np.log(a_scale)) / 3.0) - 1.0
            
            # FIXED: Isolate the fundamental scalar spectral gap eigenvalue mode (index 1)
            results[top]["x_lambda"].append(eigenvals[1])
            results[top]["y_weff"].append(w_eff)
            results[top]["z_ds"].append(compute_peak_spectral_dimension(eigenvals))

    fig = plt.figure(figsize=(9, 7), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    colors = {"Erdos-Renyi": "crimson", "Watts-Strogatz": "royalblue", "Barabasi-Albert": "darkmagenta"}
    markers = {"Erdos-Renyi": "o", "Watts-Strogatz": "s", "Barabasi-Albert": "^"}
    
    for top in topologies:
        ax.plot(results[top]["x_lambda"], results[top]["y_weff"], results[top]["z_ds"], color=colors[top], linewidth=2.5, alpha=0.8)
        ax.plot([results[top]["x_lambda"][-1]], [results[top]["y_weff"][-1]], [results[top]["z_ds"][-1]], marker=markers[top], color=colors[top], markersize=8, zorder=10)
        
    ax.set_title("The Universal 3D Renormalization Group Manifold Flow", fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel(r"Spectral Gap Scale $\lambda_1$ ($\rho_{\text{eff}}$)", fontsize=9.5, labelpad=8)
    ax.set_ylabel(r"Equation of State $w_{\text{eff}}$", fontsize=9.5, labelpad=8)
    ax.set_zlabel(r"Spectral Dimension $d_s$", fontsize=9.5, labelpad=8)
    ax.view_init(elev=22, azim=-50)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cosmological_3d_rg_manifold.png"), bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    print("--- Initializing Master Visualization Pipeline for Unified Expansion Manuscript ---")
    make_figure_1_cosmic_timeline()
    print("[1/3 Complete]: Outputted 'dynamic_cosmic_timeline_evolution.png'")
    make_figure_2_thermodynamic_stability()
    print("[2/3 Complete]: Outputted 'rupture_thermodynamic_stability.png'")
    make_figure_3_3d_rg_manifold()
    print("[3/3 Complete]: Outputted 'cosmological_3d_rg_manifold.png'")
    print("--- Master Image Asset Compilation Complete ---")
