# PAMDP / FS-HSAC formalization for multi-mode CAES plant dispatch

Status: paper §4.1–4.2 draft for **FS-HSAC v2** (feasible-support hybrid SAC).  
Code: `src/training/fs_hsac/`. Old Hybrid SAC remains an ablation (`parameterized_caes` fixed-band).

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

- \(\mathcal K(s)\subseteq\{\mathrm{dis},\mathrm{idle},\mathrm{chg}\}\) from inventory + mode lock.
- \(\mathcal M_k(s)=[\underline u_k(s),\overline u_k(s)]\) from the feasibility oracle (dynamic magnitude intervals).
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

Exact soft value / actor objectives enumerate the three modes (no Gumbel approximation):
\[
\sum_{k\in\mathcal K(s)}\pi_d(k\mid s)\,
\mathbb E_{u_k}\big[Q(s,k,u_k)-\alpha_c\log\pi_c(u_k\mid s,k)\big]
-\alpha_d\log\pi_d(k\mid s).
\]
Dual temperatures \(\alpha_d,\alpha_c\) target discrete entropy \(-\log|\mathcal K(s)|\) and continuous entropy scaled by active continuous dimension.

Critic input is \((s,u^{\mathrm{tp}},u^{\mathrm{bat}},\mathrm{onehot}(k),m)\), not a collapsed scalar \(u_{\mathrm{caes}}\).

---

## 3. Residual twin feasibility (not Bellman self-loops)

Known analytic constraints enter \(\mathcal A(s)\) as hard support. GiveSafe rejections and post-step FMU failures train a residual classifier \(C_\psi(s,a)=P(\mathrm{accept}\mid s,a)\) on a **separate** feasibility replay. They do **not** enter the economic Bellman update as fake self-loop transitions.

Actor objective may include \(\beta[-\log C_\psi(s,a)]\) once enough unsafe labels exist. GiveSafe remains the final shield (adopted, not proposed).

---

## 4. Contrast with option-SMDP / inventory HRL / old Hybrid SAC

| | OCTD3 / option-SMDP | Inventory HRL (e.g. GHTD3) | Old Hybrid SAC | FS-HSAC v2 |
|--|---------------------|----------------------------|----------------|------------|
| Decision timescale | option duration / initiation | high-level every \(c\) steps | same hour | same hour |
| Mode restriction | initiation set (can be stateful) | via goal / option choice | mask at logits | \(\mathcal K(s)\) in categorical density |
| Magnitude support | option policy | low-level continuous | static device band | \(\mathcal M_k(s)\) with Jacobians |
| Method claim | multi-timescale skills | inventory goals | hybrid heads | support-consistent hybrid SAC |

Do **not** write that options assume free mode selection. Initiation sets can restrict modes; the difference here is writing \(\mathcal K(s),\mathcal M_k(s)\) into a same-hour maximum-entropy hybrid density with exact mode enumeration.

---

## 5. Claims discipline

- PAMDP / state-dependent support is the **problem formalization**.
- Algorithm contribution is FS-HSAC (same-timescale support-consistent hybrid SAC + residual feasibility learning).
- FMI is an exchange/co-simulation standard for closed-loop verification, not a discovery.
- Disconnected legal set is a device envelope (§3.4), not a discovery.
- Dynamic intervals mix twin physics and oracle margins; sensitivity separates them.
- Do not claim FS-HSAC is universally better than HRL.
- Do not write until `docs/fs_hsac_results_gate.md` passes.
