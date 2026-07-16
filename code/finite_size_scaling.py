"""
Finite-size scaling and critical-exponent extraction for the RG-flow
attractor claimed in Section IV of the paper.

The original manuscript quoted "universal critical exponents" A=0.0166 and
B=2.5310 from a linear regression of the near-boundary RG flow, but no code
producing those numbers exists in this repository. This script:

  1. Repeats the three-topology (Erdos-Renyi, Watts-Strogatz, Barabasi-Albert)
     alpha-sweep of unified_rupture_matrix.py / generate_paper_visuals.py's
     Fig. 3 pipeline at several substrate sizes N.
  2. Checks whether the terminal fixed point (lambda_1 -> N, w_eff -> -1,
     spectral dimension d_s) is actually N-independent ("universal") or
     drifts with substrate size.
  3. Performs an explicit, reproducible power-law fit of (1 + w_eff) against
     the distance from the fixed point (1 - lambda_1/N) in the near-boundary
     region, pooling all topologies and all N, and reports the resulting
     exponent with its fit uncertainty.

This replaces the previously unsubstantiated A/B values with numbers that are
actually reproducible from code in this repo -- whatever they turn out to be.
"""
import os
import numpy as np
import scipy.linalg as la
from scipy.stats import linregress
import matplotlib.pyplot as plt

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_erdos_renyi_base(N, p=0.4, seed=42):
    rng = np.random.default_rng(seed=seed)
    adj = rng.random((N, N)) < p
    adj = np.triu(adj, 1) + np.triu(adj, 1).T
    D = np.where(adj, 1.0, 2.5)
    np.fill_diagonal(D, 0.0)
    return D


def generate_watts_strogatz_base(N, k=4, p=0.2, seed=42):
    rng = np.random.default_rng(seed=seed)
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


def generate_barabasi_albert_base(N, m=3, seed=42):
    rng = np.random.default_rng(seed=seed)
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


def run_rg_flow(N, steps=60, a_scale=1.05):
    alpha_space = np.linspace(1.5, 0.005, steps)
    topologies = ["Erdos-Renyi", "Watts-Strogatz", "Barabasi-Albert"]
    matrix_factory = {
        "Erdos-Renyi": generate_erdos_renyi_base(N),
        "Watts-Strogatz": generate_watts_strogatz_base(N),
        "Barabasi-Albert": generate_barabasi_albert_base(N, m=3),
    }
    results = {top: {"lambda1": [], "w_eff": [], "d_s": [], "alpha": []} for top in topologies}

    for alpha in alpha_space:
        for top in topologies:
            D = matrix_factory[top]
            mask = D > 0
            I = np.zeros_like(D)
            I[mask] = 1.0 / (D[mask] ** alpha)
            L_base = np.diag(np.sum(I, axis=1)) - I
            eigenvals = np.sort(la.eigvalsh(L_base))

            I_scaled = np.zeros_like(D)
            I_scaled[mask] = 1.0 / ((D[mask] * a_scale) ** alpha)
            rho_base, _ = compute_thermodynamic_metrics(eigenvals)
            rho_scaled, _ = compute_thermodynamic_metrics(
                np.sort(la.eigvalsh(np.diag(np.sum(I_scaled, axis=1)) - I_scaled))
            )
            w_eff = (-(np.log(rho_scaled / rho_base) / np.log(a_scale)) / 3.0) - 1.0

            results[top]["lambda1"].append(eigenvals[1])
            results[top]["w_eff"].append(w_eff)
            results[top]["d_s"].append(compute_peak_spectral_dimension(eigenvals))
            results[top]["alpha"].append(alpha)
    return results


def main():
    N_values = [50, 100, 200, 400]
    all_results = {N: run_rg_flow(N) for N in N_values}

    print("=====================================================================")
    print("  FINITE-SIZE SCALING OF THE TERMINAL FIXED POINT (alpha -> 0.005)")
    print("=====================================================================")
    print(f"{'N':<8}{'Topology':<18}{'lambda1_end':<14}{'lambda1_end/N':<16}{'w_eff_end':<12}{'d_s_end'}")
    print("-" * 78)
    fixed_points = []
    for N in N_values:
        for top in ["Erdos-Renyi", "Watts-Strogatz", "Barabasi-Albert"]:
            r = all_results[N][top]
            l1_end, w_end, ds_end = r["lambda1"][-1], r["w_eff"][-1], r["d_s"][-1]
            print(f"{N:<8}{top:<18}{l1_end:<14.4f}{l1_end/N:<16.4f}{w_end:<12.4f}{ds_end:.4f}")
            fixed_points.append((N, top, l1_end, w_end, ds_end))

    # -----------------------------------------------------------------
    # Pooled near-boundary power-law fit:  (1 + w_eff)  ~  C * (1 - lambda1/N)^p
    # -----------------------------------------------------------------
    x_all, y_all = [], []
    for N in N_values:
        for top in ["Erdos-Renyi", "Watts-Strogatz", "Barabasi-Albert"]:
            r = all_results[N][top]
            l1 = np.array(r["lambda1"])
            w = np.array(r["w_eff"])
            x = 1.0 - l1 / N
            y = 1.0 + w
            keep = (x > 1e-6) & (y > 1e-6)
            x_all.append(x[keep])
            y_all.append(y[keep])
    x_all = np.concatenate(x_all)
    y_all = np.concatenate(y_all)

    log_x = np.log(x_all)
    log_y = np.log(y_all)
    fit = linregress(log_x, log_y)
    exponent_p = fit.slope
    prefactor_C = np.exp(fit.intercept)

    print("\n=====================================================================")
    print("  POOLED NEAR-BOUNDARY POWER-LAW FIT: (1+w_eff) = C * (1 - lambda1/N)^p")
    print("=====================================================================")
    print(f"  p (exponent)   = {exponent_p:.4f} +/- {fit.stderr:.4f}")
    print(f"  C (prefactor)  = {prefactor_C:.4f}")
    print(f"  R^2            = {fit.rvalue**2:.4f}")
    print(f"  n pooled points = {len(x_all)}")

    # Check exponent stability across N individually (true test of "universality")
    print("\n  Per-N exponent stability check (fit within each N, pooling topologies):")
    for N in N_values:
        xN, yN = [], []
        for top in ["Erdos-Renyi", "Watts-Strogatz", "Barabasi-Albert"]:
            r = all_results[N][top]
            l1 = np.array(r["lambda1"]); w = np.array(r["w_eff"])
            x = 1.0 - l1 / N; y = 1.0 + w
            keep = (x > 1e-6) & (y > 1e-6)
            xN.append(x[keep]); yN.append(y[keep])
        xN = np.concatenate(xN); yN = np.concatenate(yN)
        fN = linregress(np.log(xN), np.log(yN))
        print(f"    N={N:<6} p = {fN.slope:.4f} +/- {fN.stderr:.4f}  (R^2={fN.rvalue**2:.4f})")

    # -----------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    colors = {"Erdos-Renyi": "crimson", "Watts-Strogatz": "royalblue", "Barabasi-Albert": "darkmagenta"}
    for N in N_values:
        for top in ["Erdos-Renyi", "Watts-Strogatz", "Barabasi-Albert"]:
            r = all_results[N][top]
            l1 = np.array(r["lambda1"]); w = np.array(r["w_eff"])
            axes[0].plot(l1 / N, w, color=colors[top], alpha=0.3 + 0.2 * N_values.index(N) / len(N_values))
    axes[0].set_xlabel(r"$\lambda_1/N$")
    axes[0].set_ylabel(r"$w_{\rm eff}$")
    axes[0].set_title("RG flow onto attractor, multiple $N$")
    axes[0].axhline(-1.0, color="black", linestyle="--", linewidth=0.8)

    axes[1].scatter(log_x, log_y, s=4, alpha=0.3, color="gray")
    xx = np.linspace(log_x.min(), log_x.max(), 50)
    axes[1].plot(xx, fit.intercept + fit.slope * xx, color="red", linewidth=2,
                 label=f"p = {exponent_p:.3f} $\\pm$ {fit.stderr:.3f}")
    axes[1].set_xlabel(r"$\ln(1-\lambda_1/N)$")
    axes[1].set_ylabel(r"$\ln(1+w_{\rm eff})$")
    axes[1].set_title("Pooled near-boundary power-law fit")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "finite_size_scaling_exponent.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved to {fig_path}")

    return exponent_p, fit.stderr, prefactor_C, fit.rvalue**2


if __name__ == "__main__":
    main()
