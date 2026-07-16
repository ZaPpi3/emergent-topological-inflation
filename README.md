# The Geometry of Information Saturation

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Official repository for the paper **"The Geometry of Information Saturation: Mapping the Complete Cosmic Expansion History from the Matrix Partition Function" (2026)** by Paul Jarvis. This is a candidate toy model in which an effective cosmic expansion history is derived from the spectral dynamics of mutual-information (MI) Laplacians on a relational quantum substrate, with the early-time de Sitter attractor following from an exact scaling identity — but the graceful-exit and late-time dark-energy mechanisms are explicit, hand-tuned phenomenological ansätze, stated as such, and a direct confrontation with real Pantheon+SH0ES supernova data is reported as a null result, including the retraction of a previously and incorrectly claimed sub-percent fit.

**[Read the paper: `main.pdf`](main.pdf)**, built from `main.tex`.

---

## 🔍 Conceptual Overview

Standard cosmology stitches together disparate components — inflaton fields early, a cosmological constant or quintessence late — to match observation, at the cost of well-known fine-tuning problems. This repository explores whether an effective expansion history can instead be read off the spectrum of a graph Laplacian built from the mutual information of a relational quantum substrate, with the macroscopic energy density defined via the canonical partition function of that spectrum. The early-time result is genuinely derived and numerically robust; the mechanisms connecting it to a realistic expansion history are honestly labeled as fitted, not derived.

## 🧠 Key Findings

- **The early-time de Sitter attractor is the strongest result here.** Under complete holographic information saturation ($\alpha \to 0^+$), the MI-Laplacian spectrum degenerates exactly, forcing $w_{\text{eff}} \to -1.0$ via an exact scaling identity (not a fit). This fixed point is verified stable to better than 0.2% across substrate sizes $N=50$–$400$ and three structurally distinct graph topologies (Erdős–Rényi, Watts–Strogatz, Barabási–Albert).
- **The graceful exit and the late-time dark-energy feature are explicit, named ansätze, not derived dynamics.** Both use a hand-fit $\alpha(t)$ profile with stated free parameters (amplitude, center, width). The paper says plainly that this is functionally equivalent to inserting a phenomenological dark-energy potential by hand — the fine-tuning problem is relocated, not solved.
- **A previously claimed <0.2% match to Pantheon+SH0ES supernova data has been retracted.** No code producing that number ever existed in this repository. A real, from-scratch fit against the public 1701-SNe Pantheon+SH0ES compilation (`code/pantheon_fit.py`) finds the model statistically indistinguishable from flat ΛCDM ($\Delta\chi^2 \approx -0.05$ for equal free parameters), with a fractional luminosity-distance deviation between the two best-fit curves reaching a mean of ~1.0% and a maximum of ~2.6% over $0<z<2.26$ — about an order of magnitude larger than the retracted figure. This is reported as a null result: consistent with the data, but not independent evidence for the model over ΛCDM.
- **A previously claimed universal spectral dimension ($d_s \approx 5.25$) was not universal.** Repeating the finite-size scaling at $N=50,100,200,400$ (`code/finite_size_scaling.py`) shows $d_s$ increases by close to 1.0 per doubling of $N$ — the old number was just the $N=100$ instance. Replaced with a reproducible, size-stable critical exponent $p = 1.11 \pm 0.02$ from a pooled power-law fit ($R^2=0.94$).
- **The $\beta \to 0^+$ equation-of-state identity is quantified rather than assumed exact at finite $\beta$.** `code/beta_invariance_break_test.py` sweeps $\beta$ over four orders of magnitude; the identity holds to within 0.77% (relative) at $\alpha=1$ for the $\beta=0.1$ used throughout the figures.

**Bottom line:** the one clean, derived result (the early-time attractor) is real and checks out under scrutiny. Everything connecting it to a full, observationally-tested expansion history is either an explicitly hand-tuned ansatz or a null result against real data. We think reporting that directly — including retracting an earlier, incorrect claim once an actual fit was performed — is more useful than a cleaner-looking headline, and matches the standard the rest of this portfolio holds itself to.

## ⚙️ The Computational Pipeline

1. **Figures 1–3 (cosmic timeline, thermodynamic stability, 3D RG manifold):** the core toy-model machinery — torus/random-graph MI-Laplacian construction, the partition-function energy density, and the $\alpha(a)$ ansätze for the graceful exit and late-time dip. *(`code/generate_paper_visuals.py`)*
2. **Standalone regenerators** for the late-time stability figure and the rupture spike, useful for iterating on the late-time ansatz independently of the full pipeline. *(`code/rupture_stability_test.py`, `code/unified_rupture_matrix.py`)*
3. **$\beta$-invariance check (Appendix A):** sweeps $\beta$ over four orders of magnitude to quantify how far the finite-$\beta$ equation of state deviates from the exact $\beta\to0^+$ identity. *(`code/beta_invariance_break_test.py`)*
4. **Finite-size scaling (Sec. V):** repeats the RG-flow attractor check at $N=50,100,200,400$ across three topologies, tests whether the spectral dimension is genuinely universal (it isn't) and fits a reproducible critical exponent. *(`code/finite_size_scaling.py`)*
5. **Pantheon+SH0ES confrontation (Sec. IV):** fits both a flat-ΛCDM baseline and the substrate-modified model to the real public supernova compilation, reports $\chi^2$/dof for both, and computes the actual fractional distance deviation between them. *(`code/pantheon_fit.py`)*

## 📁 Repository Structure

- `main.tex` / `main.pdf` : Manuscript and LaTeX source (REVTeX 4-2).
- `code/` : All six scripts above.
- `data/PantheonPlusSH0ES.dat` : The public Pantheon+SH0ES compilation (1701 SNe Ia, [DataRelease](https://github.com/PantheonPlusSH0ES/DataRelease)) — diagonal statistical+systematic errors only, not the full 1701×1701 covariance matrix, a stated simplification (see Sec. IV).
- `figures/` : All output PNGs, including the four referenced in the manuscript and three supplementary ones produced by the standalone scripts above.
- `requirements.txt` : Pinned dependency versions.

## 🔁 Reproducing

```bash
pip install -r requirements.txt
python code/generate_paper_visuals.py      # Figs. 1-3
python code/rupture_stability_test.py      # regenerates Fig. 2 standalone + console table
python code/unified_rupture_matrix.py      # regenerates the late-time rupture spike figure
python code/beta_invariance_break_test.py  # Appendix A beta-invariance table
python code/finite_size_scaling.py         # Sec. V finite-size scaling + critical exponent
python code/pantheon_fit.py                # Sec. IV Pantheon+SH0ES fit
```

All scripts use a fixed random seed (`seed=42`) where randomness is involved (Erdős–Rényi / Watts–Strogatz / Barabási–Albert graph generation), so results are deterministic across runs.

Rebuild the PDF:
```bash
tectonic main.tex
```

## ⚠️ A note on interpretation

This repository previously stated a <0.2% agreement with Pantheon+SH0ES data and a universal spectral dimension $d_s\approx5.25$; neither claim had supporting code anywhere in the repository, and both are wrong once actually computed (the real fractional distance deviation is 1-3%, and $d_s$ scales with system size rather than converging). Both are now retracted in the paper itself, with the honest numbers reported in their place, and the code that produces those honest numbers is included here. We'd rather this repository reflect that standard going forward: a claim without a script that reproduces it doesn't belong in the paper, whatever the claim.

## License

MIT - see `LICENSE`.
