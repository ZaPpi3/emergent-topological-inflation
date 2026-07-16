"""
Confront the MI-Laplacian substrate model's late-time energy-density
"rupture spike" against real Pantheon+SH0ES Type Ia supernova distances.

This script performs an honest, from-scratch fit -- it does NOT assume the
paper's claimed <0.2% agreement in advance. It:

  1. Loads the public Pantheon+SH0ES compilation (1701 SNe, data/PantheonPlusSH0ES.dat).
  2. Computes the substrate "rupture" shape g(a) = rho_eff(a)/rho_eff(1) from the
     same MI-Laplacian machinery used in unified_rupture_matrix.py (L=8 torus,
     alpha(a) profile with the late-time Gaussian dip, beta=0.1).
  3. Builds a flat-LCDM baseline model and a "substrate-modified" model in which
     the dark-energy term Omega_Lambda is replaced by Omega_Lambda * g(a).
  4. Fits both models to the real data (grid search over Om, analytic
     marginalization over the H0/absolute-magnitude nuisance offset).
  5. Reports chi2/dof for both models and the actual max/median fractional
     luminosity-distance deviation between the substrate model and the data.

Uses only diagonal statistical+systematic errors (MU_SH0ES_ERR_DIAG), not the
full 1701x1701 systematic covariance matrix -- a simplification that should be
stated as a limitation in the paper.
"""
import os
import numpy as np
import pandas as pd
import scipy.linalg as la
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

C_LIGHT = 299792.458  # km/s
DATA_PATH = os.path.join("data", "PantheonPlusSH0ES.dat")
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================================
# SUBSTRATE MODEL: rho_eff(a) SHAPE (same construction as unified_rupture_matrix.py)
# =====================================================================
def get_torus_distances(L):
    x, y = np.meshgrid(np.arange(L), np.arange(L))
    pos = np.vstack([x.ravel(), y.ravel()]).T
    N = L * L
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                dx = min(abs(pos[i, 0] - pos[j, 0]), L - abs(pos[i, 0] - pos[j, 0]))
                dy = min(abs(pos[i, 1] - pos[j, 1]), L - abs(pos[i, 1] - pos[j, 1]))
                D[i, j] = np.sqrt(dx**2 + dy**2)
    return D


def compute_thermodynamic_energy(eigenvals, beta=0.1):
    active_modes = eigenvals[1:]
    exp_weights = np.exp(-beta * active_modes)
    Z = np.sum(exp_weights)
    return np.sum(active_modes * exp_weights) / Z


def rho_eff_of_a(a, D, mask):
    ac, delta_a = 0.4, 0.05
    alpha_baseline = 4.0 * (1.0 - np.exp(-1.5 * a))
    alpha_dip = 1.8 * np.exp(-((a - ac) / delta_a) ** 2)
    alpha = max(0.005, alpha_baseline - alpha_dip)
    I = np.zeros_like(D)
    I[mask] = 1.0 / (D[mask] ** alpha)
    Lm = np.diag(np.sum(I, axis=1)) - I
    vals = np.sort(la.eigvalsh(Lm))
    return compute_thermodynamic_energy(vals)


def build_g_of_a_interpolant(L=8, n_grid=120, a_min=0.25):
    D = get_torus_distances(L)
    mask = D > 0
    a_grid = np.linspace(a_min, 1.0, n_grid)
    rho_grid = np.array([rho_eff_of_a(a, D, mask) for a in a_grid])
    g_grid = rho_grid / rho_grid[-1]  # normalize so g(a=1) = 1 today
    return interp1d(a_grid, g_grid, kind="cubic", bounds_error=False,
                     fill_value=(g_grid[0], g_grid[-1]))


# =====================================================================
# COSMOLOGICAL DISTANCES
# =====================================================================
def luminosity_distances(z_array, Om, g_interp=None, z_fine_n=4000):
    """Return D_H * D_L/D_H (i.e. D_L in units of the Hubble distance c/H0)
    on a fine grid, then interpolate onto z_array. g_interp=None -> flat LCDM."""
    z_fine = np.linspace(0.0, z_array.max() * 1.0001, z_fine_n)
    if g_interp is None:
        E = np.sqrt(Om * (1 + z_fine) ** 3 + (1 - Om))
    else:
        a_fine = 1.0 / (1 + z_fine)
        E = np.sqrt(Om * (1 + z_fine) ** 3 + (1 - Om) * g_interp(a_fine))
    inv_E = 1.0 / E
    D_C = cumulative_trapezoid(inv_E, z_fine, initial=0.0)  # comoving distance / (c/H0)
    D_C_interp = interp1d(z_fine, D_C, kind="cubic")
    D_C_at_z = D_C_interp(z_array)
    D_L_over_DH = (1 + z_array) * D_C_at_z
    return D_L_over_DH


def chi2_marginalized(mu_obs, err, D_L_over_DH):
    """mu_model = 5*log10(D_L_over_DH) + offset, offset absorbs 5log10(c/H0)+25.
    Analytically marginalize over the additive offset (standard SNe trick)."""
    mu_shape = 5.0 * np.log10(D_L_over_DH)
    resid = mu_obs - mu_shape
    w = 1.0 / err**2
    B = np.sum(resid * w)
    C = np.sum(w)
    A = np.sum(resid**2 * w)
    offset_best = B / C
    chi2_min = A - B**2 / C
    return chi2_min, offset_best


def fit_om(z, mu_obs, err, g_interp=None):
    def neg(Om):
        D_L_over_DH = luminosity_distances(z, Om, g_interp)
        chi2, _ = chi2_marginalized(mu_obs, err, D_L_over_DH)
        return chi2
    res = minimize_scalar(neg, bounds=(0.05, 0.6), method="bounded",
                           options={"xatol": 1e-4})
    Om_best = res.x
    D_L_over_DH = luminosity_distances(z, Om_best, g_interp)
    chi2_best, offset_best = chi2_marginalized(mu_obs, err, D_L_over_DH)
    return Om_best, chi2_best, offset_best


def main():
    df = pd.read_csv(DATA_PATH, sep=r"\s+")
    z = df["zHD"].to_numpy()
    mu_obs = df["MU_SH0ES"].to_numpy()
    err = df["MU_SH0ES_ERR_DIAG"].to_numpy()
    n = len(z)

    print(f"Loaded {n} SNe from Pantheon+SH0ES, z in [{z.min():.4f}, {z.max():.4f}]")

    print("Building substrate rho_eff(a) shape g(a) from MI-Laplacian model (L=8, N=64)...")
    g_interp = build_g_of_a_interpolant(L=8, n_grid=120, a_min=1.0 / (1 + z.max()) - 0.02)

    print("\nFitting flat LCDM baseline...")
    Om_lcdm, chi2_lcdm, off_lcdm = fit_om(z, mu_obs, err, g_interp=None)
    dof_lcdm = n - 2
    print(f"  Om_best = {Om_lcdm:.4f}, chi2 = {chi2_lcdm:.2f}, chi2/dof = {chi2_lcdm/dof_lcdm:.4f}")

    print("\nFitting substrate-modified model (Omega_Lambda -> Omega_Lambda * g(a))...")
    Om_sub, chi2_sub, off_sub = fit_om(z, mu_obs, err, g_interp=g_interp)
    dof_sub = n - 2
    print(f"  Om_best = {Om_sub:.4f}, chi2 = {chi2_sub:.2f}, chi2/dof = {chi2_sub/dof_sub:.4f}")

    print(f"\nDelta chi2 (substrate - LCDM) = {chi2_sub - chi2_lcdm:+.3f}  "
          f"({'substrate model favored' if chi2_sub < chi2_lcdm else 'LCDM favored'})")

    # -----------------------------------------------------------------
    # Fractional D_L deviation of substrate best-fit vs. LCDM best-fit
    # (this is the quantity closest to the paper's claimed "0.2% deviation")
    # -----------------------------------------------------------------
    z_dense = np.linspace(1e-3, z.max(), 500)
    DL_lcdm = luminosity_distances(z_dense, Om_lcdm, None) * 10 ** (off_lcdm / 5.0)
    DL_sub = luminosity_distances(z_dense, Om_sub, g_interp) * 10 ** (off_sub / 5.0)
    frac_dev = np.abs(DL_sub - DL_lcdm) / DL_lcdm * 100.0

    print(f"\nFractional D_L(z) deviation, substrate best-fit vs. LCDM best-fit:")
    print(f"  max  = {frac_dev.max():.3f}%")
    print(f"  mean = {frac_dev.mean():.3f}%")

    # Direct data residuals for both models (mu_obs - mu_model)
    DL_lcdm_at_z = luminosity_distances(z, Om_lcdm, None)
    DL_sub_at_z = luminosity_distances(z, Om_sub, g_interp)
    mu_lcdm = 5 * np.log10(DL_lcdm_at_z) + off_lcdm
    mu_sub = 5 * np.log10(DL_sub_at_z) + off_sub
    resid_lcdm = mu_obs - mu_lcdm
    resid_sub = mu_obs - mu_sub
    print(f"\nResidual scatter (mu_obs - mu_model):")
    print(f"  LCDM:      RMS = {np.sqrt(np.mean(resid_lcdm**2)):.4f} mag, max|resid| = {np.max(np.abs(resid_lcdm)):.4f} mag")
    print(f"  Substrate: RMS = {np.sqrt(np.mean(resid_sub**2)):.4f} mag, max|resid| = {np.max(np.abs(resid_sub)):.4f} mag")

    # -----------------------------------------------------------------
    # Figure: Hubble residual diagram
    # -----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7.5), dpi=300, sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    order = np.argsort(z)
    ax1.errorbar(z, mu_obs - mu_lcdm, yerr=err, fmt=".", color="gray", alpha=0.25,
                 markersize=3, elinewidth=0.5, label="Pantheon+SH0ES (rel. to LCDM)")
    ax1.plot(z[order], (mu_sub - mu_lcdm)[order], color="darkorange", linewidth=2,
             label="Substrate model best fit (rel. to LCDM)")
    ax1.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax1.set_ylabel(r"$\Delta \mu$ relative to best-fit flat $\Lambda$CDM (mag)")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_title("Substrate Rupture Model vs. Pantheon+SH0ES: Honest Residual Comparison",
                   fontsize=10.5, fontweight="bold")

    ax2.plot(z_dense, frac_dev, color="crimson", linewidth=2)
    ax2.set_xlabel("Redshift $z$")
    ax2.set_ylabel(r"$|D_L^{\rm sub} - D_L^{\rm LCDM}|/D_L^{\rm LCDM}$ (%)")
    fig.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "pantheon_fit_residuals.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved to {fig_path}")

    return {
        "n": n, "Om_lcdm": Om_lcdm, "chi2_lcdm": chi2_lcdm, "dof_lcdm": dof_lcdm,
        "Om_sub": Om_sub, "chi2_sub": chi2_sub, "dof_sub": dof_sub,
        "frac_dev_max": frac_dev.max(), "frac_dev_mean": frac_dev.mean(),
        "rms_lcdm": np.sqrt(np.mean(resid_lcdm**2)), "rms_sub": np.sqrt(np.mean(resid_sub**2)),
    }


if __name__ == "__main__":
    main()
