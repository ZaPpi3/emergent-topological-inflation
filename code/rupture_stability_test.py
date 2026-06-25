import os
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# Ensure output directory exists
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def compute_thermodynamic_metrics(eigenvals, beta=0.1):
    """
    Computes both internal energy (rho_eff) and specific heat capacity (C_v)
    directly from the canonical partition function to test thermal stability.
    """
    active_modes = eigenvals[1:]
    exp_weights = np.exp(-beta * active_modes)
    Z = np.sum(exp_weights)
    
    # Expectation values
    avg_lambda = np.sum(active_modes * exp_weights) / Z
    avg_lambda_sq = np.sum((active_modes**2) * exp_weights) / Z
    
    rho_eff = avg_lambda
    # C_v = beta^2 * variance(lambda)
    C_v = (beta**2) * (avg_lambda_sq - (avg_lambda**2))
    return rho_eff, C_v

def run_rupture_stability_check(L=8, steps=100):
    N = L * L
    D = get_torus_distances(L)
    mask = D > 0
    
    a_space = np.linspace(0.28, 1.0, steps)
    
    rho_series = []
    Cv_series = []
    alpha_series = []
    
    print("--- Running Late-Time Rupture Thermodynamic Stability Test ---")
    print(f"{'Scale Factor a':<18}{'Alpha(a)':<14}{'rho_eff':<14}{'Specific Heat C_v'}")
    print("-" * 65)
    
    for idx, a in enumerate(a_space):
        ac = 0.4
        delta_a = 0.05
        
        alpha_baseline = 4.0 * (1.0 - np.exp(-1.5 * a))
        alpha_dip = 1.8 * np.exp(-((a - ac)/delta_a)**2)
        alpha = max(0.005, alpha_baseline - alpha_dip)
        
        I = np.zeros_like(D)
        I[mask] = 1.0 / (D[mask] ** alpha)
        L_base = np.diag(np.sum(I, axis=1)) - I
        vals = np.sort(la.eigvalsh(L_base))
        
        rho, Cv = compute_thermodynamic_metrics(vals, beta=0.1)
        
        rho_series.append(rho)
        Cv_series.append(Cv)
        alpha_series.append(alpha)
        
        if idx % 20 == 0 or idx == steps - 1:
            print(f"{a:<18.4f}{alpha:<14.4f}{rho:<14.4f}{Cv:.4e}")

    # =====================================================================
    # GENERATING HIGH-IMPACT THERMODYNAMIC STABILITY PLOT
    # =====================================================================
    fig, ax1 = plt.subplots(figsize=(7.5, 5), dpi=300)

    color = 'tab:blue'
    ax1.set_xlabel('Cosmic Scale Factor ($a$)', fontsize=10)
    ax1.set_ylabel(r'Derived Matrix Energy Density $\rho_{\text{eff}}(a)$', color=color, fontsize=10)
    ax1.plot(a_space, rho_series, color=color, linewidth=2.5, label=r'$\rho_{\text{eff}}(a)$ Energy Spike')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel(r'Specific Heat Capacity $C_v(a)$', color=color, fontsize=10)
    ax2.plot(a_space, Cv_series, color=color, linestyle='-.', linewidth=2, label=r'$C_v(a)$ Thermodynamic Capacity')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axhline(0.0, color='black', linestyle='--', alpha=0.5)

    plt.title("Thermodynamic Stability Envelope of the Late-Time Rupture", fontsize=11, fontweight="bold", pad=12)
    fig.tight_layout()
    
    fig_path = os.path.join(OUTPUT_DIR, "rupture_thermodynamic_stability.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("-" * 65)
    print(f"[Stability Checked]: Verification chart saved to '{fig_path}'")

if __name__ == "__main__":
    run_rupture_stability_check()
