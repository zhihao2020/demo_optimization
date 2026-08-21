# PAMDP / FS-HSAC formalization for multi-mode CAES plant dispatch

Status: paper §4 draft for **FS-HSAC-support** (same-hour support-consistent rewrite of Hybrid SAC).  
Code: `src/training/fs_hsac/`. Live contrast: in-house fixed-band Hybrid SAC (`--method sac`, `parameterized_caes=True`). Residual $C_\psi$ is appendix-only.

---

## 1. Dispatch as a state-dependent hybrid MDP

Weekly plant dispatch is a finite-horizon MDP with hybrid actions
\[
a_t=(u^{\mathrm{tp}}_t,u^{\mathrm{bat}}_t,k_t,m_t),
\]
and a **state-dependent action support**
\[
\mathcal A(s)=
\mathcal A_{\mathrm{tp}}(s)\times
\mathcal A_{\mathrm{bat}}(s)\times
\bigcup_{k\in\mathcal K(s)}
\{k\}\times\mathcal M_k(s).
\]

- \(\mathcal K(s)\subseteq\{\mathrm{dis},\mathrm{idle},\mathrm{chg}\}\) is a mode mask.
- \(\mathcal M_k(s)=[\underline u_k(s),\overline u_k(s)]\) is an inventory interval box.
- Autoconsistency: sample and $\log\pi$ use the same \(\mathcal A(s)\). This is **not** plant/FMU feasibility (no min-run / SoC veto / FMU residual).
- Idle is a point mass (no continuous magnitude entropy).

Decoding maps normalized magnitude through the **current** physical interval (not the static device envelope alone).

---

## 2. Feasible-support hybrid policy (FS-HSAC)

\[
\pi(a\mid s)=
\pi_d(k\mid s,\mathcal K(s))\,
\pi_c(u_k\mid s,k,\mathcal M_k(s)).
\]

Continuous components use a Gaussian latent + sigmoid + affine map into \([\underline u,\overline u]\), with log-density corrected by both Jacobians. Illegal modes receive \(-\infty\) logits.

Actor / critic objectives use a discrete sum over \(\mathcal K(s)\) (no Gumbel approximation). That sum may be exact; do **not** write exact hybrid entropy:
\[
\sum_{k\in\mathcal K(s)}\pi_d(k\mid s)\,
\mathbb E_{u_k}\big[Q(s,k,u_k)-\alpha_c\log\pi_c(u_k\mid s,k)\big]
-\alpha_d\log\pi_d(k\mid s).
\]
Dual temperatures \(\alpha_d,\alpha_c\) target discrete entropy \(-\log|\mathcal K(s)|\) and continuous entropy scaled by active continuous dimension.

Critic input is \((s,u^{\mathrm{tp}},u^{\mathrm{bat}},\mathrm{onehot}(k),m)\), not a collapsed scalar \(u_{\mathrm{caes}}\).

---

## 3. Residual twin feasibility (appendix only)

GiveSafe is adopted on the live stack. Residual \(C_\psi\) (split feasibility replay, optional \(\beta[-\log C_\psi]\)) is an **appendix** variant of full FS-HSAC, not the live claim. Rejections do **not** enter the economic Bellman update as fake self-loop transitions.

---

## 4. Contrast with option-SMDP / inventory HRL / old Hybrid SAC

| | OCTD3 / option-SMDP | Inventory HRL (e.g. GHTD3) | Old Hybrid SAC | FS-HSAC v2 |
|--|---------------------|----------------------------|----------------|------------|
| Decision timescale | option duration / initiation | high-level every \(c\) steps | same hour | same hour |
| Mode restriction | initiation set (can be stateful) | via goal / option choice | mask at logits | \(\mathcal K(s)\) in categorical density |
| Magnitude support | option policy | low-level continuous | static device band | \(\mathcal M_k(s)\) with Jacobians |
| Method claim | multi-timescale skills | inventory goals | hybrid heads | support-consistent hybrid SAC |

Do **not** write option/HRL back into the contribution. OCTD3 did not compare MILP; do not write RL>exact opt. GHTD3's convex QP often has the lowest scalar CC because the surrogate is low-dimensional.

---

## 5. Claims discipline

- **One contribution:** same-hour Hybrid SAC ties sampling and $\log\pi$ to \(\mathcal A(s)=\mathcal K(s)\times\mathcal M_k(s)\).
- Contrast: in-house fixed-band Hybrid SAC (latent density then clamp).
- FMI / TOU / ETS / disconnected CAES envelope / electricity-only sale are **setting**.
- GiveSafe is adopted.
- Do not claim superiority while `docs/fs_hsac_results_gate.md` is false.
- Do not claim FS-HSAC is universally better than HRL.
- Do not write until `docs/fs_hsac_results_gate.md` passes.
