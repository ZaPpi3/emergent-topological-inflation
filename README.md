# Emergent Topological Inflation: Relational Matrix Engine

This repository hosts the master numerical simulation ecosystem for a field-free, information-theoretic cosmology. By evaluating the canonical partition function of Mutual Information (MI) Laplacians on relational quantum substrates, this engine derives the complete expansion history of the universe—from the initial non-singular Big Bang bounce and de Sitter inflation attractors to late-time dark energy anomalies—as a seamless sequence of stable thermodynamic phase transitions.

## 📁 Repository Architecture

```text
├── code/
│   └── generate_paper_visuals.py      # Master simulation pipeline for manuscript figures
├── figures/                           # Publication-grade PNG vector field assets
│   ├── dynamic_cosmic_timeline_evolution.png  # Figure 1: Field-free expansion trajectory
│   ├── rupture_thermodynamic_stability.png   # Figure 2: Specific heat capacity envelope
│   └── cosmological_3d_rg_manifold.png       # Figure 3: Multi-topology 3D RG manifold flow
├── main.tex                           # Flagship single-column LaTeX manuscript source
├── main.pdf                           # Compiled preprint letter manuscript
└── LICENSE                            # MIT Open-Source License
```

## ⚙️ Installation & System Requirements

The processing core relies on Python 3.9+ alongside standard scientific computing and matrix diagonalization libraries. 

Clone the repository and install the verified dependencies using your preferred package manager:

```bash
git clone https://github.com
cd emergent-topological-inflation
pip install numpy scipy matplotlib
```

## 🚀 Simulation Pipelines

All mathematical engines are consolidated inside a single, high-fidelity script. Running it handles the matrix allocations, centered finite-difference stencils, and heat-kernel random walk calculations, exporting the results directly into your `figures/` directory.

To trigger the master compilation suite:
```bash
python code/generate_paper_visuals.py
```

### Generated Figure Insights

1. **`dynamic_cosmic_timeline_evolution.png` (Figure 1)**: Trajectories tracking the scale factor $a(t)$. It demonstrates that as space expands, the substrate naturally localizes, providing an automatic "graceful exit" off the inflation floor ($w = -1.0$) toward the radiation-dominated boundary ($w = 1/3$).
2. **`rupture_thermodynamic_stability.png` (Figure 2)**: Tracks the late-time dark energy re-entanglement anomaly centered at $a_c \approx 0.4$. It logs the specific heat capacity ($C_v$) to numerically verify that the system remains strictly positive ($C_v > 0$) and stable throughout the transition.
3. **`cosmological_3d_rg_manifold.png` (Figure 3)**: Sweeps non-spatial network microstates (Erdős–Rényi, Watts–Strogatz, and Barabási–Albert architectures), proving that widely separated structural layouts collapse onto a unique, universal 3D Renormalization Group manifold tracking toward a stable spectral dimension of $d_s \approx 5.25$.

## 🧠 Core Physics & Theoretical Framework

Rather than manually injecting phenomenological scalar fields or finely tuned dark energy potentials into the stress-energy tensor, this model derives the macroscopic cosmic expansion from first-principles statistical mechanics. 

By treating the eigenvalues $\lambda_n$ of the combinatorial MI-Laplacian matrix as the foundational spectrum of the relational quantum network's structural fluctuations, the effective cosmic energy density $\rho_{\text{eff}}$ is extracted natively from the canonical partition function:

$$Z = \sum_{n=1}^{N-1} e^{-\beta \lambda_n}, \quad \rho_{\text{eff}} = -\frac{\partial \ln Z}{\partial \beta} = \frac{\sum \lambda_n e^{-\beta \lambda_n}}{\sum e^{-\beta \lambda_n}}$$

where $\beta^{-1}$ functions strictly as a microscopic modular entanglement temperature (grounded in Tomita–Takesaki vacuum localization theory) which acts as the renormalization group coarse-graining energy scale. 

* **Early-Universe Singularity Regularization**: At the absolute information-saturation limit ($\alpha \rightarrow 0^+$), the spectrum becomes perfectly degenerate ($\lambda_n = N$). Plugging this flat spectrum into the partition formula cancels the temperature scaling weights completely, mathematically forcing the energy density to lock onto a hard algebraic ceiling ($\rho_{\text{eff}} = N$), bypassing the unphysical classical general relativistic singularity ($\rho \rightarrow \infty$).
* **Late-Time Observational Viability**: Integrating the resulting modified Friedmann equation under the secondary re-entanglement network transition yields a late-time transient dark energy spike. Confronted directly against the precision envelope of the Pantheon+ Type Ia supernova compilation, the model bounds maximum absolute residual distance deviations below $0.2\%$, rendering it structurally continuous with standard $\Lambda$CDM under current observational limits.

## 📜 Open-Source Licensing

This research software engine is licensed under the open-source **MIT License**. Feel free to deploy, modify, and distribute the computational pipelines with appropriate scientific attribution.
