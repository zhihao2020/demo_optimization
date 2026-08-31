# PAMDP / PC-HybridTD3 formalization for multi-mode CAES plant dispatch

文档更新：2026-08-30 22:40 (+08:00)

Status: paper §4 draft for **PC-HybridTD3** (physics-constrained parameterized hybrid TD3).  
Code: `src/training/hybrid_td3/` with `parameterized_caes=True` and dynamic \(\mathcal A_f(s)\).  
Live entry: `scripts/train_seasonal.py --method td3`. Projection ablation: `--ablation projection`. Static-support ablation: `--ablation static-support`. FS-HSAC remains in `src/training/fs_hsac/` as archive.

---

## 1. Dispatch as a state-dependent hybrid MDP

Weekly plant dispatch is a finite-horizon MDP with hybrid actions
\[
a_t=(u^{\mathrm{th}}_t,u^{\mathrm{bat}}_t,m_t,z_t),
\qquad m_t\in\{D,I,C\},\quad z_t\in[0,1],
\]
and a **state-dependent action support**
\[
\mathcal A_f(s)=
\mathcal A_{\mathrm{tp}}(s)\times
\mathcal A_{\mathrm{bat}}(s)\times
\bigcup_{k\in\mathcal K(s)}
\{k\}\times\mathcal M_k(s).
\]

- \(\mathcal K(s)\subseteq\{\mathrm{dis},\mathrm{idle},\mathrm{chg}\}\) is a mode mask from FeasibilityOracle.
- \(\mathcal M_k(s)=[\underline u_k(s),\overline u_k(s)]\) is the current magnitude interval (twin physics plus oracle margins).
- Device boxes \(\mathcal A_T\times\mathcal A_B\times\mathcal A_C\) are **not** \(\mathcal A_f(s)\). Interchange couples the commands:
\[
\mathcal A_f(s)=(\mathcal A_T\times\mathcal A_B\times\mathcal A_C)\cap\mathcal A_{\mathrm{grid}}.
\]
Analytic decoder: CAES interval \(\cap\) grid window (drop empty modes) \(\to u_C\) \(\to\) tighten thermal \(\to u_T\) \(\to\) conditional battery. GiveSafe is a residual screen; greedy evaluation tries the actor **once**.
- The actor must satisfy \(a_t\in\mathcal A_f(s_t)\), **not** \(a_t=\Pi_{\mathcal A_f}(\pi(s_t))\) and **not** a cartesian product followed by 64 GiveSafe draws.
- Idle is a point mass.

Decoding (the paper's core formula):
\[
u^{\mathrm{CAES}}=
\begin{cases}
\underline u_D(s)+z(\overline u_D(s)-\underline u_D(s)), & m=D,\\
0, & m=I,\\
\underline u_C(s)+z(\overline u_C(s)-\underline u_C(s)), & m=C.
\end{cases}
\]

---

## 2. Physics-constrained hybrid TD3

Actor: mode logits + one magnitude head; Straight-Through Gumbel \(\tau:1.0\to 0.2\) on the actor update; evaluation and the TD3 target use \(\arg\max m\).

Critic: \(Q(s,u^{\mathrm{th}},u^{\mathrm{bat}},e_m,z)\) (action dim 6). Do not score a collapsed scalar \(u_{\mathrm{caes}}\) on \([-1,1]\).

Target:
1. \(m'=\arg\max_m \ell_{\bar\theta}(s')\) (no Gumbel, no mode noise)
2. \(\tilde z'=\mathrm{clip}(z'+\varepsilon,0,1)\) with clipped Gaussian \(\varepsilon\)
3. re-decode with **next-state** intervals \(\mathcal M_{m'}(s')\)

Economic replay \(\mathcal D_B\): physical FMU transitions only (`physical_fraction=1.0`). GiveSafe rejections go to a safety audit set and never the Bellman update.

Trainer knobs aligned with Cui 2024 Table 4 where they transfer: \(\varepsilon:1.0\to 0.05\), \(\mathrm{lr}=10^{-4}\), \(\tau=0.005\), batch \(64\), \(\gamma=0.99\). TD3 delay \(D=2\). Warm-up \(N_{\mathrm{warm}}=1024\) random-feasible physical hours; stop if \(N_D\le 100\) or \(N_C\le 100\). `storage_use.enabled: false`. Terminal SOC \(\xi=0.06\); report \(E_{\mathrm{terminal}}\) as well as the bonus.

---

## 3. GiveSafe (adopted)

GiveSafe is the last screen before the FMU steps (fallback off, 64 tries). Residual \(C_\psi\) is **not** part of the PC-HybridTD3 claim (that was FS-HSAC appendix).

---

## 4. Contrast with option-SMDP / inventory HRL / FS-HSAC

| | OCTD3 / option-SMDP | Inventory HRL (e.g. GHTD3) | FS-HSAC (archive) | PC-HybridTD3 |
|--|---------------------|----------------------------|-------------------|--------------|
| Decision timescale | option duration / initiation | high-level every \(c\) steps | same hour | same hour |
| Mode restriction | initiation set | via goal / option | \(\mathcal K(s)\) in SAC density | \(\mathcal K(s)\) + \(\arg\max\) target |
| Magnitude support | option policy | low-level continuous | \(\mathcal M_k(s)\) + Jacobian | \(\mathcal M_k(s)\) affine decode |
| Method claim | multi-timescale skills | inventory goals | support-consistent hybrid SAC | physics-constrained hybrid TD3 |

Do **not** write option/HRL or SAC entropy back into the contribution. Do not write RL > MILP.

---

## 5. Claims discipline

- **One contribution:** same-hour parameterized hybrid TD3 whose actor/critic live on \(\mathcal A_f(s)\), with physical-only economic Bellman.
- Two ablations only: continuous vs hybrid; static vs dynamic support.
- Four main methods: rule, rolling MILP, projection TD3, PC-HybridTD3.
- Week protocol: 36 train / 8 val / 8 test. Tables = TEST. 8760 h = deployment.
- Empty tables until Stage D. No `fs_hsac_*` cash, no `seasonal_v1` year-constant TOU.
