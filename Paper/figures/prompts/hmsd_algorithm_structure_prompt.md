# HMSD Algorithm Structure Figure Prompt

适用类型：框架图 / 算法架构图  
配色方案：A — Okabe-Ito 学术标准  
推荐分辨率：16:9 或 3:2，2K  

---

## English prompt (for NanoBanana / Gemini / academic-figure-generator)

```
A highly detailed, information-dense academic paper algorithm architecture diagram in the style of top-tier Applied Energy / IEEE PES journal figures. The diagram illustrates Hierarchical Market-Safe Dispatch (HMSD) for a thermal–battery–CAES multi-energy plant under TOU tariffs and ETS carbon cost, arranged as a two-band layout: upper AGENT band and lower ENVIRONMENT band, with solid arrows for training updates and dashed arrows for c-step interaction and feedback. White-dominated flat technical illustration, no 3D glossy effects, no photorealism, no decorative clip art.

=== GLOBAL TITLE ===
Centered top title bar in small-caps Steel Blue text: "Hierarchical Market-Safe Dispatch (HMSD)". Subtitle in medium grey: "Agent–Environment architecture and hierarchical TD3 policy learning".

=== SECTION AGENT (upper ~62% of figure) ===
Large dashed rounded panel with faint grey #F7F7F7 fill and thin Light Grey #CCCCCC border. Small-caps italic Steel Blue label at top-left: "Agent".

--- Sub-panel: c-Step Rolling Interaction and Experience Collection ---
Top-left white rounded box with Steel Blue border labeled "c-Step Rolling Interaction and Experience Collection Process". Inside: two side-by-side buffer modules:
1) High-level Replay Buffer D^hi: orange stack icons of transitions, min-batch arrows, caption "(s_t, g_t, R^ext_{t:t+c-1}, s_{t+c}, tau) each c steps".
2) Low-level Replay Buffer D^lo: orange stack icons, caption "(s_i, g_i, a_i, r^int_i, s_{i+1}, g_{i+1}) each hour".
Above buffers, two grey mechanism tags:
- "MS-HER / Each c steps" (Steel Blue border)
- "F-MLE / Warm-start Initialization" (Warm Orange border)

--- Sub-panel: Historical Goal Relabeling with MS-HER ---
Left-middle lavender-tinted very faint panel (#F7F7F7 with thin Steel Blue border) titled "Historical Goal Relabeling with MS-HER". Content boxes:
- Original goal g_t
- Candidate pool C_t: original / achieved Δs^int / future goals / optional market prior
- Formula in monospace: g̃_t ∈ argmax_{g∈C_t} Σ w_i^act log π_lo(a_i | s_i, g_i)
- Relabeled transition card: (s_t, g̃_t, R^ext, s_{t+c})
Dashed arrow from D^hi into MS-HER; solid arrow from MS-HER back into high-level TD3 update.

--- Sub-panel: High-level Controller: TD3 with MSGP ---
Right-upper light green-tinted faint panel (#F0F7F4) with Bluish Green #009E73 border. Title "High-level Controller: TD3 with MSGP".
Inside left: Actor Network schematic (3-layer MLP thumbnail) with μ_φ^hi(s) → g ∈ R^5, goal vector components labeled [Δbat, Δgas, Δth, u_tp, arb].
Inside below actor: MSGP blend box: g ← (1−w_m−w_r) μ^hi(s) + w_m g^mkt + w_r g^rec with TOU valley-charge / peak-discharge prior and recovery prior H=40–48 h.
Inside right: Twin Critic Networks Q_θ1, Q_θ2 schematic; loss min_θ (Q − y)^2 with y = R^hi + γ^c min_k Q̄(s', g').
Output arrow labeled g_t downward to low-level.

--- Sub-panel: Low-level Controller: TD3 with GiveSafe ---
Right-middle warm sand-tinted faint panel with Warm Orange #E69F00 border. Title "Low-level Controller: TD3 with GiveSafe".
Inside left: Actor Network π_lo(s_n, κ g) with κ=4, s_n = tanh(s/s0), hybrid action heads u_tp, u_battery, caes_mode(3), caes_magnitude.
Inside center: GiveSafe projection box with Vermillion accent border: a* = Π_{F(s)}(ã), formula a_t^* = Π_{F(s_t)}(π_lo(s_n, κ g) + noise).
Inside right: Twin Critics Q(s, g, a); target with policy smoothing.
Solid arrow a_t^* downward into Environment.

=== SECTION ENVIRONMENT (lower ~38% of figure) ===
Large dashed panel faint grey background, small-caps italic Steel Blue label "Environment".

--- Left: Uncertainty / Boundary Factors ---
Four small monochrome line-chart thumbnails in a 2×2 grid, each white box with grey border:
TOU price, Wind, PV/irradiance, Load. Caption "Exogenous boundaries (price-taker TOU + resources)".

--- Center: System model ---
White box "Sysplorer Modelica FMU twin (1 h step)": three device icons in a row — Thermal unit, Battery, CAES (compressor / thermal storage / expander) — labeled "thermal–battery–CAES". Caption "FMI closed-loop plant".

--- Right: Rewards and costs ---
Purple-bordered faint panel "Comprehensive monetary feedback":
r^ext = r^econ + r^shape + r^term
with r^econ from Δcash − π_CO2 Δm_CO2 − C^CUT(curt+unserved) − C^deg
r^int = −||e|| + α r^ext
Also note carbon ETS π_CO2=80 CNY/t and unserved/curtailment penalties.
Dashed feedback arrows upward: s_{t+1}, r^ext, r^int → Agent buffers and controllers.

=== GLOBAL ANNOTATIONS ===
Legend at bottom center:
- Solid dark-grey arrow: "Algorithm training process (TD3 updates)"
- Dashed dark-grey arrow: "Interaction process at each c-step interval (c=8)"
- Dotted amber arrow: "F-MLE warm-start (before RL only)"
Small note bottom-right: "Only executable a* ∈ F(s) enter physical replay; MS-HER relabels high-level goals only."

=== STYLE SPECIFICATIONS ===
Okabe-Ito academic palette: Steel Blue #0072B2 for primary module borders and section labels; Warm Orange #E69F00 for secondary / GiveSafe emphasis; Bluish Green #009E73 for high-level success path (sparse); Vermillion #D55E00 only for GiveSafe projection accent; module fills Pure White #FFFFFF; region backgrounds Faint Grey #F7F7F7; standard borders Light Grey #CCCCCC; body text Charcoal #333333; arrows Dark Grey #4D4D4D; secondary notes Medium Grey #666666.
Flat 2D vector-like academic diagram, high information density, readable 10pt-equivalent labels, consistent rounded rectangles, no rainbow fills, no gradient backgrounds, no 3D shadows, no cartoon mascots, no watermark, no fake photographs of equipment, grayscale-print safe, white paper background, journal figure quality suitable for full-width single-column landscape placement.
SPELL EXACTLY these strings: HMSD, MS-HER, F-MLE, MSGP, GiveSafe, TD3, D^hi, D^lo, Sysplorer FMU, TOU, CAES.
```
