# Vesicle Electrostriction Dashboard Walkthrough (v6)
This document describes the design and biophysical models implemented in the
**Vesicle Electrostriction Dashboard v6** (`app.py`), which simulates
the results of:
> **"Electrostrictive Forces on Vesicles with Compartmentalized Permittivity and Conductivity Conditions"** (Rey et al., IEEE TDEI, 2009).

---

## Changelog

| Version | Key Changes |
|---------|-------------|
| **v6** | Start of V6 development. Branding and documentation updates. |
| **v5** | Single-iframe custom component architecture (declare_component) completely eliminating flashing. Client-side animation loop and instant parameter sync. |
| **v4** | Impedance corrected to solid-sphere CM factor (aligned to f_MW). Force-proportional deformation. Theme via `config.toml`. |
| v3 | Slanted T1/T2/T3 phase boundary. Full precomputed cache. Permittivity-ratio mode. |
| v2 | Prolate to oblate physics fixed. Hyuga/Gao validation. |
| v1 | Initial prototype. |

---

## 1. Physical and Mathematical Models

### A. Single-shell Maxwell–Wagner coefficients
For a spherical vesicle (core + thin capacitive/conductive membrane) in an
external field, with complex conductivities $\sigma^{\ast} = \sigma + j\omega\epsilon_0\epsilon$
and membrane sheet admittance $Y_m = (\sigma_m + j\omega\epsilon_0\epsilon_m)/\delta$:

$$\sigma^{\ast}_{\rm eff} = \frac{Y_m R\,\sigma^{\ast}_i}{Y_m R + \sigma^{\ast}_i}, \qquad
f_{\rm CM} = \frac{\sigma^{\ast}_{\rm eff} - \sigma^{\ast}_o}{\sigma^{\ast}_{\rm eff} + 2\sigma^{\ast}_o}$$

The **dipole field amplitude** $a = f_{\rm CM} E_0$ and **interior uniform field**
$b = \dfrac{3\sigma^{\ast}_o}{\sigma^{\ast}_{\rm eff} + 2\sigma^{\ast}_o} E_0$ are bounded (order $E_0$).

### B. Maxwell stress coefficients
From the time-averaged Maxwell stress tensor jump, the surface pressure is

$$p(\theta) = K_{r,c}\cos^2\theta + K_{r,s}\sin^2\theta + K_\theta\cos\theta\sin\theta$$

with $K_{r,c}, K_{r,s}, K_\theta$ computed in `stress_coefficients(a,b,E_0,eps_in,eps_out)`.
Both the integrated forces and the 3D surface pressure derive from this one function.

### C. Resultant forces with frequency dependence (Hyuga / Gao model)
The pole and equator resultants use the bracket evaluated at $\theta=0$ and
$\theta=\pi/2$, with the conductivity ratio blended toward unity across the
Maxwell-Wagner crossover $f_c = \dfrac{\sigma_{in}+2\sigma_{out}}{2\pi\epsilon_0(\epsilon_{in}+2\epsilon_{out})}$:

$$\alpha(f) = \frac{1}{1+(f/f_c)^2}, \quad R_{s,b} = 1 + \alpha(R_s - 1), \quad R_{\epsilon,b}=R_\epsilon$$

$$FR_z \propto (R_{s,b}^2 - R_{\epsilon,b}), \qquad FR_r \propto (R_{\epsilon,b} - 1)$$

**The slant.** The crossover $f_c$ depends on the ratio because $\sigma_{out}=\sigma_{in}/R_s$.
As $R_s$ increases, $f_c$ falls, so the prolate/oblate to sphere boundary **slants**
in the (frequency, ratio) plane — reproducing the Fig. 7a behaviour.

**Three regions (Fig. 7a).**
$T_1$ = sphere to prolate for $R_s>1$;
$T_2$ = sphere to oblate for $R_s<1$;
$T_3$ = low-frequency prolate/oblate toggle.

**Validated sign rules:**
* $\epsilon_{in}=\epsilon_{out}$: $\sigma_{in}/\sigma_{out} > 1 \Rightarrow$ **prolate**; $< 1 \Rightarrow$ **oblate** (Fig. 5, 7a).
* $\sigma_{in}=\sigma_{out}$: $\epsilon_{in}/\epsilon_{out} < 1 \Rightarrow$ **prolate**; $> 1 \Rightarrow$ **oblate** — reversed (Fig. 6, 7b).

### D. Impedance — v4 correction

> **v3 issue:** The lumped RC model placed the impedance transition at the *membrane RC frequency*
> (~36 Hz) — three orders of magnitude below the Maxwell-Wagner force transition (~225 kHz).

**v4 fix — solid-sphere Clausius-Mossotti factor:**

$$K_{\rm CM}(\omega) = \frac{\sigma^{\ast}_{in} - \sigma^{\ast}_{out}}{\sigma^{\ast}_{in} + 2\sigma^{\ast}_{out}}, \qquad \sigma^{\ast} = \sigma + j\omega\epsilon_0\epsilon$$

$$\sigma^{\ast}_{\rm eff}(\omega) = \sigma^{\ast}_{out}(1 + 3K_{\rm CM}), \qquad Z(\omega) = \frac{1}{\sigma^{\ast}_{\rm eff} \cdot 4\pi R}$$

$K_{\rm CM}$ has a single Debye dispersion at exactly $f_{\rm MW}$, identical to $f_c$
in the Hyuga force model. The impedance step in $Z_{\rm real}$ and peak in $|Z_{\rm imag}|$
now sit at the same frequency as the prolate-to-sphere force crossover, as in the paper.

**Why the shelled-sphere model failed:** The `effective_coefficients` function includes
the membrane as a separate layer with admittance $Y_m$. Its characteristic frequency
is the *membrane* RC time constant, not $f_{\rm MW}$. The solid-sphere CM factor
bypasses the membrane layer (consistent with the Hyuga approximation) and relaxes
at the Maxwell-Wagner frequency.

### E. Vesicle deformation — v4 (force-proportional)

The aspect ratio $k$ now scales **continuously** with the normalised force difference:

$$k = 1 + k_{\rm max} \cdot \frac{F_z - F_r}{\max|F_z - F_r|}$$

where $k_{\rm max} \approx 0.7$. Shape tracks the physical pressure anisotropy:
near-spherical where forces are small, maximum deformation where forces are large.

### F. Maxwell-Wagner relaxation frequency

$$f_{\rm MW} = \frac{\sigma_{in}+2\sigma_{out}}{2\pi\epsilon_0(\epsilon_{in}+2\epsilon_{out})}$$

For the paper's values this lands at approximately $2.2 \times 10^5$ Hz ($\sigma_{out}=7.5$ uS/cm)
and $5.6 \times 10^5$ Hz ($\sigma_{out}=30$ uS/cm), matching the reported inflection frequencies.

---

## 2. Interactive Dashboard Layout

```mermaid
graph TD
    A[app.py v6] --> B[1. 2D Pseudo-3D Vesicle Shape and Stress (Animated)]
    A --> C[2. Phase Diagram - cond. or perm.]
    A --> D[3. Frequency Sweep - Forces and Impedance]
    A --> V[4. Validation Panel - Hyuga/Gao/model and self-tests]

    subgraph Controls
        E[Sweep variable: Conductivity or Permittivity]
        F[Ratio select_slider]
        G[Frequency select_slider]
        H[Fixed Parameters]
        I[Animate Ratio Sweep toggle]
        J[Animate Frequency Sweep toggle]
    end

    E --> A
    F --> A
    G --> A
    I --> A
    J --> A
    C -- Click --> F
    C -- Click --> G
    D -- Click --> G
```

### Components
1. **Sidebar**: sweep-variable toggle, ratio and frequency select-sliders (1 Hz to 1 GHz),
   toggles to animate the ratio or frequency sweep (both ON for a 4-Quadrant closed sweep loop),
   and the fixed paper-accurate constants. Sliders are disabled when their corresponding sweep is animated.
2. **2D Pseudo-3D Shape and Stress (Plot 1)**: tilted wireframe showing parallels, meridians, and electric field lines, colored by local Maxwell stress.
   Animate modes sweep and loop client-side endlessly.
3. **Phase Diagram (Plot 2)**: heatmap of shape class over frequency x ratio. The transition
   edge slants — reproducing Fig. 7a/7b. The three regions T1/T2/T3 are annotated.
4. **Frequency Sweep (Plot 3)**: $FR_z$ (red), $FR_r$ (blue), $Z_{\rm real}$ (green dotted),
   $Z_{\rm imag}$ (green dashed) vs frequency. Impedance transition aligned with force
   crossover (v4 fix). Clicking sets the frequency.
5. **Validation Panel (Plot 4)**: overlays Hyuga, Gao, and model stress; reports $f_{\rm MW}$;
   runs 7 automated self-tests live.

---

## 3. Automated Self-Tests

* Dipole amplitude bounded ($|a| < 5E_0$).
* Conductivity sign flip: $FR_z(R_s=2)>0$ and $FR_z(R_s=0.5)<0$.
* Permittivity sign flip (reversed): $FR_z(R_\epsilon=0.5)>0$ and $FR_z(R_\epsilon=2)<0$.
* $f_{\rm MW} \in [10^5, 10^6]$ Hz for the paper's conductivities.
* Hyuga and Gao share an identical angular shape.
* Slanted boundary: prolate-to-sphere edge frequency at ratio = 100 is lower than at ratio = 2.
* Sphere-band consistency: at 50 MHz (ratio = 2) the force difference stays within the threshold.

All seven pass in the current build.

---

## 4. How to Run
```bash
streamlit run app.py --server.port 8502
```
Then open `http://localhost:8502`.

---

## 5. Code vs. Paper: Agreements & Disagreements

A detailed item-by-item comparison of the simulator against Rey et al. (2009) is maintained in:

> [`science/code_vs_paper_report.md`](code_vs_paper_report.md)

Summary: 7 quantitative agreements, 4 approximations, 4 items not implemented (COMSOL FEM, membrane RC in forces, osmotic coupling, frequency-dependent material properties).
