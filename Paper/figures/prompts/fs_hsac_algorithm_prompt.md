# FS-HSAC algorithm figure — Image-model prompt (Applied Energy)

Style D classic accent bar. Three generation attempts; pick by label/arrow fidelity.

## Canonical labels (spell exactly)

- FS-HSAC
- Sysplorer FMU, 1 h
- A(s) = K(s) x M_k(s)
- inference: sample chosen mode only
- GiveSafe (adopted, fallback off)
- D_B physical Bellman
- D_F rejections / C_psi only
- exact 3-mode V(s) at update

## Prompt (attempt base)

Create a classic academic accent-bar technical architecture diagram for an Applied Energy journal paper. The diagram is a same-hour closed loop for feasible-support hybrid SAC (FS-HSAC) on a Sysplorer Modelica FMU twin. Flat, grayscale-safe, publication quality. No decoration, no 3D, no photos, no icons.

VISUAL STYLE — CLASSIC ACCENT BAR:
- Horizontal section bands stacked vertically on pale gray #F7F7F5
- Each section has a thick 8 px colored LEFT ACCENT BAR
- Content boxes: white fill, thin #DDDDDD border, 4 px rounded corners
- Sans-serif Helvetica/Arial, bold titles, regular body
- Colored arrows match their SOURCE section
- Clean, flat, zero decoration, generous whitespace
- Landscape 16:9, full-width two-column journal figure

COLOR PALETTE — OCEAN DUSK:
- Deep teal #264653 for text and FMU band
- Teal #2A9D8F for actor / support
- Gold #E9C46A for GiveSafe
- Sandy orange #F4A261 for split replay
- Burnt coral #E76F51 for reject path only
- Background #F7F7F5, box fill #FFFFFF

LAYOUT — FOUR HORIZONTAL BANDS, top to bottom:

BAND 1 (teal bar) title "TWIN":
Left box title "Sysplorer FMU, 1 h" body "thermal-BESS-CAES DAE"
Right box title "state s" body "SoC, thermo, power, monthly TOU"

BAND 2 (teal bar) title "SUPPORT + ACTOR":
Three boxes left to right:
1. "oracle" / "A(s) = K(s) x M_k(s)"
2. "FS-HSAC actor" / "masked mode head; discharge and charge magnitude heads; thermal and battery Gaussians"
3. "inference: sample chosen mode only"

BAND 3 (gold bar) title "SCREEN":
One wide box "GiveSafe (adopted, fallback off)" / "N_try = 64; only accepted hours step the FMU"

BAND 4 (orange bar) title "REPLAY + UPDATE":
Three boxes:
1. "D_B physical Bellman" / "accepted (s, a, r, s')"
2. "D_F rejections / C_psi only" / "never a Bellman self-loop"
3. "exact 3-mode V(s) at update" / "twin Q; dual alpha_d, alpha_c; C_psi penalty not a second hard gate"

CONNECTIONS:
1. state s -> oracle, solid teal, label "build A(s)"
2. oracle -> FS-HSAC actor, solid teal
3. FS-HSAC actor -> inference box, solid teal
4. inference -> GiveSafe, solid gold, label "decoded (k, mu)"
5. GiveSafe ACCEPT solid teal down into Sysplorer FMU, label "step 1 h"
6. GiveSafe REJECT dashed burnt coral #E76F51 into D_F, label "reject"
7. FMU accept -> D_B, solid orange, label "physical transition"
8. D_B and D_F -> exact 3-mode V(s) at update, solid slate
9. update thin gray arrow back up to FS-HSAC actor, label "theta"
10. FMU next-state arrow back to state s, solid teal, label "next s"

CONSTRAINTS:
- SPELL EVERY LABEL EXACTLY as written above
- Do not write HMSD, hierarchical, c-step, option, high-level goal
- Do not write that rejected transitions enter replay or Bellman
- No figure number, no caption, no watermark, no logo
- FMI is the exchange standard, not a novelty badge
- No photorealistic plant, no clip art
