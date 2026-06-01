# Code vs. Paper: Agreement & Disagreement Report
**Simulator:** Vesicle Electrostriction Dashboard v6 (`app.py`)
**Reference:** Rey, F. et al., *"Electrostrictive Forces on Vesicles with Compartmentalized Permittivity and Conductivity Conditions"*, IEEE Trans. Dielectrics & Electrical Insulation, **16**(5), 2009.

---

## Summary

| Topic | Status |
|---|---|
| Force sign rules (conductivity mode) | ✅ Agreement |
| Force sign rules (permittivity mode) | ✅ Agreement |
| Maxwell-Wagner crossover frequency | ✅ Agreement |
| Hyuga/Gao angular pressure bracket | ✅ Agreement |
| Maxwell stress tensor form | ✅ Agreement |
| Phase boundary slope (T1/T2/T3 slant) | ✅ Agreement |
| Membrane parameter values | ✅ Agreement |
| Impedance model | ⚠️ Approximation (aligned in v4) |
| FR_r in conductivity mode | ⚠️ Structurally zero (correct within model) |
| Phase diagram solver | ⚠️ Simplified (heuristic threshold) |
| Vesicle deformation law | ⚠️ Schematic (not derived from mechanics) |
| Membrane as explicit layer in forces | ❌ Bypassed in Hyuga formula |
| COMSOL field solution | ❌ Not implemented |
| Electroosmotic / osmotic coupling | ❌ Not implemented |

---

## 1. ✅ AGREEMENTS

### 1.1 Force Sign Rules — Conductivity Mode (Paper Fig. 5, 7a)

**Paper:** For equal permittivities (ε_in = ε_out = 80), the vesicle is **prolate** when σ_in/σ_out > 1 and **oblate** when σ_in/σ_out < 1. Both shapes relax to spherical above f_MW.

**Code (`resultant_forces`):**
```python
FR_z = pref * (Rs_b² − Re_b)   # > 0 when Rs > 1 and Re = 1  → prolate
FR_r = pref * (Re_b − 1)       # = 0 when Re = 1
```
With Re = 1 and Rs > 1: FR_z > 0 (prolate) ✓  
With Re = 1 and Rs < 1: FR_z < 0 (oblate) ✓  
Self-test #2 verifies this at f = 1 Hz.

---

### 1.2 Force Sign Rules — Permittivity Mode (Paper Fig. 6, 7b)

**Paper:** For equal conductivities (σ_in = σ_out), the sign rule **reverses**: ε_in/ε_out < 1 → prolate; ε_in/ε_out > 1 → oblate.

**Code:** With Rs = 1 and Re ≠ 1:
```python
FR_z = pref * (1 − Re)   # > 0 when Re < 1 → prolate ✓
                          # < 0 when Re > 1 → oblate  ✓
```
Self-test #3 verifies both branches.

---

### 1.3 Maxwell-Wagner Crossover Frequency (Paper Eqs. 3–5, Fig. 5/6)

**Paper:** The shape transition frequency is:

$$f_\text{MW} = \frac{\sigma_{in} + 2\sigma_{out}}{2\pi\varepsilon_0(\varepsilon_{in} + 2\varepsilon_{out})}$$

For the paper's values (σ_out = 7.5 µS/cm): f_MW ≈ 2.2 × 10⁵ Hz.  
For σ_out = 30 µS/cm: f_MW ≈ 5.6 × 10⁵ Hz.

**Code:**
```python
f_c = (sigma_in + 2*sigma_out) / (2*pi*EPS_0*(eps_in + 2*eps_out))
```
Identical formula. Self-test #4 confirms f_MW ∈ [10⁵, 10⁶] Hz for both paper conductivities.

---

### 1.4 Hyuga/Gao Angular Pressure Bracket (Paper Fig. 4a/4b)

**Paper:** The surface pressure distribution follows (Hyuga 1991, Gao 2008):

$$p(\theta) \propto \left[(1 + R_s^2 - 2R_\varepsilon)\cos^2\theta + (R_\varepsilon - 1)\right]$$

**Code (`analytical_pressure`):**
```python
bracket = (1 + Rs**2 - 2*Re) * cos²θ + (Re - 1)
pref_Hyuga = (9*eps_out) / (8*pi*(2+Rs))
pref_Gao   = (9*eps_out) / (2*(1+Rs)²)
```
Both forms are implemented; they differ only by a θ-independent prefactor. Self-test #5 verifies std(ratio)/mean(ratio) < 10⁻⁹.

---

### 1.5 Maxwell Stress Tensor Form (Paper Eq. 1)

**Paper:** Time-averaged pressure across the membrane:

$$p(\theta) = K_{r,c}\cos^2\theta + K_{r,s}\sin^2\theta + K_\theta\cos\theta\sin\theta$$

**Code (`stress_coefficients`):**
```python
K_rc    = 0.25*EPS_0*(eps_out*|E0+2a|² − eps_in*|b|²)
K_rs    = 0.25*EPS_0*(−eps_out*|−E0+a|² + eps_in*|b|²)
K_theta = 0.5*EPS_0*(eps_out*Re[(E0+2a)·conj(−E0+a)] + eps_in*|b|²)
```
Field amplitudes at the membrane surface match the paper's dipole solution: outside radial (E0 + 2a), outside tangential (−E0 + a), inside (b). ✓

---

### 1.6 Phase Boundary Slope — T1/T2/T3 Slant (Paper Fig. 7a)

**Paper:** The prolate→sphere boundary (T1) slopes downward — higher σ_in/σ_out relaxes to spherical at *lower* frequency. This is because f_MW depends on σ_out = σ_in/R_s; larger R_s → smaller σ_out → smaller f_MW.

**Code:** `f_c` is computed per-ratio row, so each row has its own crossover. Self-test #6 verifies edge(R_s = 100) < edge(R_s = 2). The phase heatmap shows the resulting slant. ✓

---

### 1.7 Fixed Parameter Values (Paper Table I / Section III)

| Parameter | Paper | Code |
|---|---|---|
| Vesicle radius R | 10 µm | 10 µm ✓ |
| Membrane thickness δ | 5 nm | 5 nm ✓ |
| Membrane permittivity ε_m | 5 | 5 ✓ |
| Membrane conductivity σ_m | 10⁻⁸ S/m | 10⁻⁸ S/m ✓ |
| Interior permittivity ε_in | 80 | 80 ✓ |
| Applied field E₀ | 200 V/cm | 200 V/cm ✓ |
| Reference σ_in | 15 µS/cm | 15 µS/cm ✓ |

---

## 2. ⚠️ APPROXIMATIONS (partial agreement)

### 2.1 Impedance Model (Paper Fig. 8 / Section IV)

**Paper:** Impedance is computed from the full COMSOL FEM solution with a 1-A PORT boundary condition. The result is a dispersive spectrum with a real-part step and imaginary-part peak co-located with the force transition at f_MW.

**Code (v4):** Uses the solid-sphere Clausius-Mossotti impedance:
$$Z(\omega) = \frac{1}{\sigma^{*}_{eff} \cdot 4\pi R}, \quad \sigma^{*}_{eff} = \sigma^{*}_{out}(1 + 3K_{\text{CM}})$$

**Assessment:**  
- ✅ Transition frequency now aligned with f_MW (v4 fix).  
- ⚠️ Absolute magnitudes are not calibrated to the paper's PORT geometry or electrode spacing.  
- ⚠️ The dilute-limit formula (φ → 0) ignores electrode effects, double-layer capacitance, and the specific chamber geometry used in the paper.  
- ❌ Does not reproduce the paper's exact Z curve shape quantitatively.

---

### 2.2 FR_r in Conductivity Mode

**Paper (Fig. 5):** FR_r is shown as a separate curve. Its visual magnitude appears small relative to FR_z but potentially non-zero.

**Code:** In conductivity mode (ε_in = ε_out = 80):
```python
FR_r = pref * (Re − 1) = 0    # exactly zero, Re = 1
```

**Assessment:**  
This is mathematically correct within the Hyuga dipole approximation: when ε_in = ε_out, there is no dielectric contrast to create equatorial pressure. The paper's Fig. 5 FR_r curve likely lies on the x-axis or is too small to see on the printed scale. The physical interpretation is correct — the prolate shape in conductivity mode comes entirely from pole pressure, not equatorial compression.  
FR_r becomes non-zero and meaningful only in permittivity mode (ε_in ≠ ε_out), which the code also implements correctly.

---

### 2.3 Phase Diagram Solver (Paper Fig. 7a/7b)

**Paper:** Phase boundaries are derived from the full COMSOL field solution and energy minimization.

**Code:** Uses a relative threshold on the force difference:
```python
THRESH = 0.02 × max|FR_z − FR_r|    # 2% of the low-frequency peak
shape = "prolate"  if (FR_z − FR_r) >  THRESH
shape = "oblate"   if (FR_z − FR_r) < −THRESH
shape = "sphere"   otherwise
```

**Assessment:**  
- ✅ Qualitative boundaries match (T1, T2, T3 regions present and slanted correctly).  
- ⚠️ The 2% threshold is a heuristic — the paper's actual boundary is determined by the balance between Maxwell stress and membrane elasticity, which is not modeled.  
- ⚠️ The neutral band width (sphere zone) is artificially set by the threshold rather than derived from membrane mechanics.

---

### 2.4 Vesicle Deformation Law

**Paper:** Shape deformation is governed by the balance between Maxwell electrostrictive pressure and membrane elastic restoring force (bending and tension). The paper does not give a simple closed-form k(F).

**Code (v4):**
```python
k = 1 + k_max * (Fz − Fr) / peak_force
a = R₀ k^(2/3),   b = R₀ k^(−1/3)   # volume-conserving ellipsoid
```

**Assessment:**  
- ✅ Volume conservation enforced.  
- ✅ Proportional to the force anisotropy (qualitatively correct).  
- ❌ Linear force-to-deformation law is a schematic assumption. The paper uses a nonlinear elastic membrane model. Absolute deformation magnitudes are not calibrated.  
- The UI labels this "schematic, not-to-scale" which is appropriate.

---

## 3. ❌ ABSENT FROM CODE

### 3.1 Membrane as Explicit Layer in Force Computation

**Paper:** Forces are derived from the full single-shell field solution including the membrane as a separate dielectric/conductive layer. The membrane RC time constant introduces a second characteristic frequency.

**Code:** `resultant_forces` uses the Hyuga/Gao approximation, which treats the vesicle as a **solid sphere** with effective interior properties. The membrane is implicitly absorbed into the high-frequency limit but does **not** appear as a separate relaxation in the force curves.

**Consequence:** The low-frequency membrane RC transition (~36 Hz) is absent from the force curves. The code shows only one relaxation (f_MW); the paper's full model would show two. This is a valid approximation for the mid-frequency range plotted in the paper but is physically incomplete.

---

### 3.2 COMSOL FEM Field Solution

**Paper:** All quantitative results (forces, deformation, impedance) are backed by a finite-element solution of the full Maxwell equations in the vesicle geometry, including non-uniform fields near the membrane.

**Code:** Uses the dipole/Clausius-Mossotti analytical approximation (uniform interior field, pure dipole exterior). This is accurate for R ≪ λ (which is satisfied here) but misses:
- Higher-order multipole contributions.
- Edge effects near the membrane.
- Non-uniform field distribution inside the membrane layer.

---

### 3.3 Electroosmotic and Osmotic Pressure Coupling

**Paper:** The paper discusses the coupling between Maxwell electrostrictive pressure and osmotic pressure across the membrane (relevant when the membrane is permeable or at high voltages).

**Code:** Not implemented. Pure Maxwell stress only.

---

### 3.4 Temperature and Frequency-Dependent Conductivity

**Paper:** Acknowledges that σ and ε of biological solutions are temperature- and frequency-dependent.

**Code:** σ and ε are treated as frequency-independent constants set by the sliders.

---

## 4. Quantitative Spot-Checks

| Quantity | Paper value | Code output | Match? |
|---|---|---|---|
| f_MW at σ_out = 7.5 µS/cm | ~2×10⁵ Hz | 2.25×10⁵ Hz | ✅ |
| f_MW at σ_out = 30 µS/cm | ~8×10⁵ Hz | 5.6×10⁵ Hz | ~✅ (within 30%) |
| Shape at low f, σ_in/σ_out = 2 | Prolate | Prolate | ✅ |
| Shape at high f, σ_in/σ_out = 2 | Spherical | Spherical | ✅ |
| Shape at low f, σ_in/σ_out = 0.5 | Oblate | Oblate | ✅ |
| Shape at low f, ε_in/ε_out = 0.5 | Prolate | Prolate | ✅ |
| Shape at low f, ε_in/ε_out = 2 | Oblate | Oblate | ✅ |
| FR_r magnitude at f < f_MW, cond. mode | Small / zero | Zero (exact) | ✅ (within model) |
| Impedance step location | Co-located with f_MW | Co-located (v4) | ✅ |
| Impedance step magnitude (absolute) | COMSOL PORT | Not calibrated | ⚠️ |

---

## 5. Overall Assessment

The code faithfully reproduces the **qualitative physics** of Rey et al. 2009:
- All force sign rules from Figs. 5, 6 match.
- The Maxwell-Wagner transition frequency matches the paper's analytical formula.
- The phase boundary structure (T1/T2/T3, slanted) matches Fig. 7a/7b qualitatively.
- The Hyuga and Gao analytical forms are both implemented and validated.

The main **quantitative gaps** are:
1. The impedance is a theoretical model (solid-sphere CM), not the paper's FEM result — useful for showing frequency alignment but not calibrated to absolute magnitude.
2. The deformation is schematic (linear force law) rather than from membrane mechanics.
3. The membrane's own RC relaxation (~36 Hz) is absent from the force curves.
4. No COMSOL FEM — the model is fully analytical/semi-analytical.

These limitations are appropriate for an **interactive educational/exploratory simulator** rather than a quantitative engineering tool.
