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
Dual temperatures \(\alpha_d,\alpha_c\) use Haarnoja's \(L(\alpha)=-\alpha(\mathbb E[\log\pi]+H_{\mathrm{target}})\). Discrete target is \(0.98\log|\mathcal K(s)|\) (Christodoulou), **not** \(-\log|\mathcal K|\). Continuous target is \(-\mathbb E[\mathrm{cont\_dim}]\) on \(\mathbb E[\log\pi_c]\), not analytic Gaussian entropy.

Trainer knobs that transfer from Cui 2024 Table 4 (OCTD3): \(\varepsilon_{\max}=1.0\), \(\varepsilon_{\min}=0.05\), \(\Delta\varepsilon=6\times10^{-6}\) on a \(2\times10^5\)-step run, \(\alpha_{\mathrm{lr}}=10^{-4}\), \(\tau=0.005\), batch \(64\), \(\gamma=0.99\), replay \(10^4\). The fair week is \(5000\times168=8.4\times10^5\) steps, so \(\varepsilon\) and replay are scaled to the same horizon fraction ( \(\varepsilon\) hits \(0.05\) at \(158\,333/200\,000\approx0.79\) of training; replay \(4.2\times10^4\) ). TD3 delay \(D=2\), Gaussian \(\sigma^2=0.1\), option temperature \(\kappa=1\), and PFI/CCI weight \(w=0.2\) are **not** SAC knobs and are not copied. Dual temperatures stay clamped to \([0.05,2]\) (Haarnoja, not Cui \(\kappa\)). `storage_use` is Eq. (35) \(R^F\) on \(r^{\mathrm{ext}}\) only; mix weight \(0.5\). \(\theta=50\) MW is a plant-scale stand-in: Cui's \(\theta_{\mathrm{thr}}\) is “from historical data”, no numeric value.

文档更新：2026-08-30 20:20 (+08:00)

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
