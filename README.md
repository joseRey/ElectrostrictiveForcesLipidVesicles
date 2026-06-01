# Vesicle Electrostriction Simulator

An interactive Streamlit dashboard simulating time-averaged AC electrostrictive forces
and Maxwell stress distributions on lipid vesicles under an AC electric field.

---

## Primary Reference (COMSOL / FEM model)

> **[1] Rey, F., Rentschler, M., Boehm, H., Leize-Wagner, E., & Wagner, A. (2009).**
> *"Electrostrictive Forces on Vesicles with Compartmentalized Permittivity and Conductivity Conditions."*
> **IEEE Transactions on Dielectrics and Electrical Insulation, 16(5), 1344–1353.**
> DOI: [10.1109/TDEI.2009.5293966](https://doi.org/10.1109/TDEI.2009.5293966)
>
> This is the primary reference for all results. The paper uses COMSOL Multiphysics FEM
> to compute the full Maxwell stress tensor, vesicle deformation, and impedance spectra
> for single-shell spherical vesicles under AC fields.

---

## How This Simulator Relates to the Paper

> **The paper [1] uses a full COMSOL Multiphysics FEM solution.**
> This simulator uses analytical approximations to reproduce the same physics.

### Analytical model used

The paper computes forces, deformation, and impedance by solving the full Maxwell equations
numerically for a single-shell spherical vesicle.  Because a full FEM solver is not
available in an interactive Python dashboard, this simulator instead uses two well-known
analytical approximations that the paper itself validates:

| Model | Reference | What it provides |
|-------|-----------|-----------------|
| **Hyuga quasi-static force formula** | Hyuga, Kinosita & Wakabayashi [2] | Frequency-dependent pole (FR_z) and equatorial (FR_r) resultant forces via a Debye-relaxation blend of conductivity and permittivity ratios across the Maxwell-Wagner crossover |
| **Gao alternative prefactor** | Gao, Feng, Yin & Gao [3] | Same angular bracket as Hyuga but different scaling — used for cross-validation in the Validation Panel (Plot 4) |
| **Solid-sphere CM impedance** | Maxwell [4], Wagner [5] | Complex impedance Z(ω) via the Clausius-Mossotti factor; transition co-located with f_MW |

The analytical model correctly reproduces the **qualitative** behaviour of the FEM results:
prolate/oblate sign rules, the slanted T1/T2/T3 phase boundaries (Fig 7a/7b), and the
co-location of the impedance inflection with the Maxwell-Wagner force crossover.

**What is NOT reproduced:** the absolute magnitudes of deformation (which require membrane
elasticity), the exact impedance curve shape (which requires the electrode geometry), and the
membrane RC relaxation (~36 Hz) that would appear as a second feature in the full FEM impedance.

### Full agreement and disagreement analysis

See [science/code_vs_paper_report.md](science/code_vs_paper_report.md) for an item-by-item
comparison of every figure and result from the paper against this simulator's output:
7 quantitative agreements, 4 approximations, 4 absent features.

---

## All References Used in This Model

| # | Reference | Role in simulator |
|---|-----------|-------------------|
| **[1]** | Rey et al., IEEE TDEI, 2009 | Primary: all figures, parameter values, phase diagrams (Figs 5–8) |
| **[2]** | Hyuga, Kinosita & Wakabayashi, Bioelectrochem. Bioenerg., 1991 | Analytical force formula & angular pressure bracket (Fig 4) |
| **[3]** | Gao, Feng, Yin & Gao, J. Mech. Phys. Solids, 2008 | Alternative prefactor for same angular bracket (Fig 4 validation) |
| **[4]** | Maxwell, J. C., *A Treatise on Electricity and Magnetism*, 1873 | Maxwell stress tensor foundation |
| **[5]** | Wagner, K. W., Arch. Elektrotech., 1914 | Maxwell-Wagner interfacial polarisation theory |

### Full Citations

**[2]** Hyuga, H., Kinosita Jr., K., & Wakabayashi, N. (1991).
*"Transient and steady-state electrodeformation of biological cells."*
Bioelectrochemistry and Bioenergetics, **26**(1), 101–111.
DOI: [10.1016/0302-4598(91)87054-I](https://doi.org/10.1016/0302-4598(91)87054-I)

> Derives the DC/quasi-static analytical formula for Maxwell electrostrictive pressure
> on a single-shell vesicle:
> `p(theta) ~ (9 eps_out)/(8 pi (2+Rs)) * E0^2 * [(1+Rs^2-2Re)cos^2(theta)+(Re-1)]`

**[3]** Gao, L.-T., Feng, X.-Q., Yin, Y.-J., & Gao, H. (2008).
*"An electromechanical liquid crystal model of biological cells."*
Journal of the Mechanics and Physics of Solids, **56**(9), 2844–2862.
DOI: [10.1016/j.jmps.2008.04.006](https://doi.org/10.1016/j.jmps.2008.04.006)

> Provides an equivalent angular bracket with a different prefactor:
> `p ~ (9 eps_out)/(2*(1+Rs)^2) * E0^2 * bracket`
> Both Hyuga and Gao forms are implemented and cross-validated in the simulator
> (Validation Panel, Plot 4). They share an identical angular distribution
> and differ only by a theta-independent scaling constant.

---

## Code vs. Paper: Agreements and Disagreements

See [`science/code_vs_paper_report.md`](science/code_vs_paper_report.md) for the full analysis. Summary:

### Agreements (7)

| Item | Paper | Simulator |
|------|-------|-----------|
| Force sign, conductivity mode | sigma_in > sigma_out -> prolate (Fig 5, 7a) | Reproduced via FR_z = pref*(Rs_b^2 - Re_b) |
| Force sign, permittivity mode | eps_in > eps_out -> oblate (Fig 6, 7b, reversed) | Reproduced; sign flip when Rs=1 |
| Maxwell-Wagner crossover | f_MW = (si+2so)/(2*pi*e0*(ei+2eo)) | Identical formula; verified in self-test |
| Angular pressure bracket | Hyuga (1991), Gao (2008) bracket form | Both implemented; std(ratio)/mean < 1e-9 |
| Maxwell stress tensor form | p(theta) = K_rc cos^2 + K_rs sin^2 + K_theta sin*cos | Exact match |
| Phase boundary slant (Fig 7a) | Higher ratio -> lower crossover frequency | f_c computed per-ratio; slant confirmed |
| Fixed parameter values (Table I) | R=10um, delta=5nm, eps_m=5, E0=200V/cm | All matched |

### Approximations (4)

| Item | Paper | Simulator | Impact |
|------|-------|-----------|--------|
| Impedance model | COMSOL FEM with PORT BC | Solid-sphere CM factor (v4) | Transition frequency aligned to f_MW; absolute magnitude not calibrated |
| FR_r (conductivity mode) | Small non-zero curve in Fig 5 | Exactly zero (Re=1 -> no equatorial contrast) | Negligible; physically correct within Hyuga approximation |
| Phase boundary | Energy minimization + FEM | Heuristic 2% force-difference threshold | Boundary shape qualitatively correct; width uncalibrated |
| Vesicle deformation | Nonlinear membrane elastic model | Linear force-proportional scaling | Schematic; volume conserved; not quantitative |

### Not Implemented (4)

| Item | Reason |
|------|--------|
| COMSOL FEM field solution | Would require finite-element solver; beyond scope of analytical dashboard |
| Membrane RC relaxation in forces | Hyuga approximation bypasses the membrane as a separate layer |
| Electroosmotic / osmotic pressure | Only Maxwell stress included |
| Frequency/temperature-dependent conductivity | Parameters treated as constants |

---

## Features

* **3D Vesicle Simulator and Stress Distribution (Plot 1)**: volume-conserving ellipsoidal
  deformation with force-proportional aspect ratio and full Maxwell stress colormap.
* **Interactive Phase Diagram (Plot 2)**: heatmap over 1 Hz to 1 GHz and ratio 0.01 to 100;
  reproduces Fig 7a/7b slanted T1/T2/T3 boundaries.
* **Frequency Sweep (Plot 3)**: FR_z, FR_r, Z_real, Z_imag vs frequency; impedance
  transition aligned with f_MW (v4); interactive click sets the operating point.
* **Validation Panel (Plot 4)**: Hyuga, Gao, and model angular stress overlay (Fig 4);
  f_MW report; 7 automated self-tests.

---

## Version History

| Version | Key Changes |
|---------|-------------|
| **v4** (current) | Impedance corrected to solid-sphere CM factor aligned to f_MW. Force-proportional deformation. |
| v3 | Slanted T1/T2/T3 phase boundary. Full precomputed cache for O(1) slider lookups. Permittivity-ratio mode. |
| v2 | Prolate to oblate transition physics fixed. Hyuga/Gao validation panel. |
| v1 | Initial prototype. |

---

## Repository Structure

```text
├── README.md
├── app.py                                             # Dashboard (v6)
├── science/
│   ├── 2009_electrostrictive_forces_vesicles_Rey_Gilbert.pdf
│   ├── vesicle_electrostriction_walkthrough.md        # Physics walkthrough (v6)
│   └── code_vs_paper_report.md                       # Agreements & disagreements
└── .streamlit/
    └── config.toml
```

---

## Scientific Core (Key Equations)

### Maxwell-Wagner crossover and force resultants (Hyuga [2])

$$f_c = \frac{\sigma_{in}+2\sigma_{out}}{2\pi\epsilon_0(\epsilon_{in}+2\epsilon_{out})}, \qquad \alpha(f) = \frac{1}{1+(f/f_c)^2}$$

$$FR_z \propto R_{s,b}^2 - R_{\epsilon,b}, \qquad FR_r \propto R_{\epsilon,b} - 1$$

### Impedance — solid-sphere CM factor (v4, aligned to f_MW)

$$K_{\rm CM}(\omega) = \frac{\sigma^{\ast}_{in} - \sigma^{\ast}_{out}}{\sigma^{\ast}_{in} + 2\sigma^{\ast}_{out}}, \quad \sigma^{\ast} = \sigma + j\omega\epsilon_0\epsilon$$

$$Z(\omega) = \frac{1}{\sigma^{\ast}_{\rm eff} \cdot 4\pi R}, \quad \sigma^{\ast}_{\rm eff} = \sigma^{\ast}_{out}(1 + 3K_{\rm CM})$$

### Vesicle deformation — force-proportional (v4)

$$k = 1 + k_{\rm max} \cdot \frac{F_z - F_r}{\max|F_z - F_r|}, \qquad a = R_0 k^{2/3}, \quad b = R_0 k^{-1/3}$$

---

## Installation and Usage

```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8502
```
Open `http://localhost:8502`.
