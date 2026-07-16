import numpy as np
import scipy.linalg as la

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

def compute_thermodynamic_energy(eigenvals, beta):
    active_modes = eigenvals[1:]
    exp_weights = np.exp(-beta * active_modes)
    Z = np.sum(exp_weights)
    return np.sum(active_modes * exp_weights) / Z

def run_thermal_break_test(L=8):
    N = L * L
    D = get_torus_distances(L)
    mask = D > 0
    
    # Extreme temperature/scale sweep across 4 orders of magnitude
    beta_sweep = [0.001, 0.01, 0.1, 1.0, 10.0]
    a_scale = 1.05
    
    print("=====================================================================")
    print("        MI-LAPLACIAN THERMAL ATTRACTOR BREAK-TEST PROFILE            ")
    print("=====================================================================")
    print(f"{'Beta (β)':<12}{'α=0 Ceiling':<18}{'α=1.0 w_eff':<16}{'EoS Stability'}")
    print("-" * 68)
    
    # 1. Compute Base Spectra once for optimization
    # Saturated State (alpha = 0)
    I_sat_base = np.zeros_like(D); I_sat_base[mask] = 1.0
    vals_sat_base = np.sort(la.eigvalsh(np.diag(np.sum(I_sat_base, axis=1)) - I_sat_base))
    
    I_sat_scaled = np.zeros_like(D); I_sat_scaled[mask] = 1.0
    vals_sat_scaled = np.sort(la.eigvalsh(np.diag(np.sum(I_sat_scaled, axis=1)) - I_sat_scaled))
    
    # Ordinary State (alpha = 1.0)
    I_ord_base = np.zeros_like(D); I_ord_base[mask] = 1.0 / D[mask]
    vals_ord_base = np.sort(la.eigvalsh(np.diag(np.sum(I_ord_base, axis=1)) - I_ord_base))
    
    I_ord_scaled = np.zeros_like(D); I_ord_scaled[mask] = 1.0 / (D[mask] * a_scale)
    vals_ord_scaled = np.sort(la.eigvalsh(np.diag(np.sum(I_ord_scaled, axis=1)) - I_ord_scaled))

    for beta in beta_sweep:
        # Extract saturated boundary metrics
        rho_sat_base = compute_thermodynamic_energy(vals_sat_base, beta)
        
        # Extract ordinary state fluid metric scaling
        rho_ord_base = compute_thermodynamic_energy(vals_ord_base, beta)
        rho_ord_scaled = compute_thermodynamic_energy(vals_ord_scaled, beta)
        
        scaling_exponent = np.log(rho_ord_scaled / rho_ord_base) / np.log(a_scale)
        w_eff_ord = (-scaling_exponent / 3.0) - 1.0
        
        # Calculate deviation from the analytical centerpiece identity: w = 1/3 - 1 = -0.6667
        eos_error = abs(w_eff_ord - (-0.6666666666666666))
        
        print(f"{beta:<12.3f}{rho_sat_base:<18.4f}{w_eff_ord:<16.4f}{eos_error:.4e}")
    print("=====================================================================")

if __name__ == "__main__":
    run_thermal_break_test(L=8)
