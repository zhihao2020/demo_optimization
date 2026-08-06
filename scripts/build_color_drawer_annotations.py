#!/usr/bin/env python
"""Create labelled redraw guides for the Image-2 concept bases.

The guides deliberately keep the generated illustration untouched and add
checked English labels as a separate vector layer.  They are handoff material
for Color Drawer, not replacements for the paper figures.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Paper" / "figures" / "image2_sources" / "raw"
OUT = ROOT / "Paper" / "figures" / "color_drawer_annotations"

COL = {
    "ink": "#172B4D", "blue": "#0B63CE", "teal": "#008A8A",
    "orange": "#D96416", "purple": "#6D3CB8", "green": "#25835A",
    "red": "#B53A4B", "gray": "#425466",
}


def canvas(source):
    image = plt.imread(source)
    h, w = image.shape[:2]
    fig, ax = plt.subplots(figsize=(16, 16 * h / w), dpi=170)
    ax.imshow(image, extent=(0, 100, 0, 100), origin="upper")
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")
    return fig, ax


def note(ax, xy, text, color="ink", fs=8.4, ha="center"):
    ax.text(
        *xy, text, ha=ha, va="center", fontsize=fs, color=COL[color],
        linespacing=1.12, fontweight="semibold",
        bbox={"boxstyle": "round,pad=0.28,rounding_size=0.12", "fc": "white",
              "ec": COL[color], "lw": 0.85, "alpha": 0.96}, zorder=6,
    )


def arrow(ax, start, end, color="ink", dashed=False):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=9, lw=1.05,
        linestyle="--" if dashed else "-", color=COL[color], zorder=5,
    ))


def heading(ax, text):
    ax.text(1.8, 98.5, text, ha="left", va="top", fontsize=11.5,
            fontweight="bold", color=COL["ink"],
            bbox={"boxstyle": "round,pad=0.24", "fc": "white", "ec": "none", "alpha": 0.93},
            zorder=8)


def write(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=260, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig1():
    fig, ax = canvas(RAW / "fig1_system_base_v1.png")
    heading(ax, "Fig. 1 redraw guide - multi-energy plant and FMU twin")
    note(ax, (8, 80), "SUPPLY LAYER", "blue", 9.3)
    note(ax, (21, 74), "Grid interface\nTOU buy / sell", "blue")
    note(ax, (63, 79), "Wind + PV\nrenewable supply", "teal")
    note(ax, (8, 56), "CONVERSION LAYER", "orange", 9.3)
    note(ax, (19, 49), "Thermal unit\n$u_{\\mathrm{tp}}$", "orange")
    note(ax, (52, 48), "CAES compressor-expander\nmode $m_{\\mathrm{caes}}$, magnitude $\\mu_{\\mathrm{caes}}$", "blue", 7.7)
    note(ax, (8, 32), "STORAGE LAYER", "purple", 9.3)
    note(ax, (20, 27), "Battery\n$u_{\\mathrm{bat}}$, SoC", "blue")
    note(ax, (55, 27), "CAES inventories\ngas / hot / cold", "orange")
    note(ax, (37, 7.8), "Demand / market layer: electric load + grid settlement", "gray")
    note(ax, (87.5, 54), "Modelica-FMU twin\nstate / command loop, 1 h step", "teal", 8.1)
    note(ax, (83, 13), "Solid: electric / thermal / gas flow\nDashed: observation / control", "gray", 7.4)
    arrow(ax, (83, 15), (89, 45), "gray", True)
    write(fig, "fig1_color_drawer_annotated")


def fig4():
    fig, ax = canvas(RAW / "fig4_reward_safety_base_v2.png")
    heading(ax, "Fig. 4 redraw guide - reward composition and safety projection")
    note(ax, (12, 84), "Economic reward\n$r_t^{\\mathrm{econ}}=\\Delta J_t/C_{\\mathrm{ref}}$", "blue", 7.7)
    note(ax, (14, 55), "SoC potential shaping\n$r_t^{\\mathrm{shape}}=\\kappa_t(L_{1,t-1}-L_{1,t})-\\kappa_t^{\\mathrm{abs}}L_{1,t}$", "teal", 6.7)
    note(ax, (14, 31), "Terminal recovery gate\n$r_T^{\\mathrm{term}}=B$ if $L_1^e\\leq\\varepsilon$; otherwise $-pL_1^e$", "purple", 6.5)
    note(ax, (47, 75), "Reward synthesis (three terms are summed)\n$r_t^{\\mathrm{econ}}+r_t^{\\mathrm{shape}}+r_t^{\\mathrm{term}}\\longrightarrow r_t^{\\mathrm{ext}}$", "ink", 7.2)
    note(ax, (72, 88), "High-level return\n$R_t^{\\mathrm{hi}}=c^{-1}\\sum_{j=0}^{c-1}r_{t+j}^{\\mathrm{ext}}$", "orange", 7.2)
    note(ax, (50, 48), "Low-level TD3 receives $r_i^{\\mathrm{int}}$\n$r_i^{\\mathrm{int}}=-\\|e_i\\|_w+\\alpha r_i^{\\mathrm{ext}}$", "ink", 7.0)
    note(ax, (50, 39), "Raw hybrid action $\\tilde a_i$", "orange", 7.2)
    note(ax, (50, 27), "GiveSafe projection\n$a_i=\\Pi_{\\mathcal{F}(s_i)}(\\tilde a_i)$", "teal", 7.1)
    note(ax, (80, 21), "Modelica-FMU transition\n$(s_i,a_i)\\rightarrow(s_{i+1},\\Delta J_i)$\npost-step hard checks", "gray", 6.8)
    note(ax, (82, 48), "Feedback to reward terms:\n$\\Delta J_i$, inventories, terminal state", "gray", 6.6)
    arrow(ax, (80, 47), (57, 72), "gray", True)
    arrow(ax, (78, 22), (82, 31), "gray")
    write(fig, "fig4_color_drawer_annotated")


def fig5():
    fig, ax = canvas(RAW / "fig5_hierarchical_rl_base_v1.png")
    heading(ax, "Fig. 5 redraw guide - Safe Market-GHTD3 closed loop")
    note(ax, (8.5, 61), "High-level TD3\n$\\mu^{\\mathrm{hi}}(s_t)$", "blue", 7.8)
    note(ax, (25, 61), "MSGP\nmarket / recovery prior", "blue", 7.5)
    note(ax, (50, 61), "Absolute 5-D goal every $c$ steps\n$g_t=[\\Delta\\mathrm{bat},\\Delta\\mathrm{gas},\\Delta\\mathrm{th},u_{\\mathrm{tp}},\\mathrm{arb}]$", "purple", 6.5)
    note(ax, (76.5, 61), "Low-level TD3\n$\\pi_{\\mathrm{lo}}(s_i,\\kappa g_i)$", "green", 7.7)
    note(ax, (23, 31), "Observations $s_i$\nprices, resources, load,\npower and inventories", "blue", 6.7)
    note(ax, (43, 29), "F-MLE warm-start\nfeasible rule trajectories", "orange", 6.7)
    note(ax, (61, 28), "Integrated energy environment\nthermal + battery + CAES + grid", "teal", 6.7)
    note(ax, (78, 28), "GiveSafe\n$\\Pi_{\\mathcal{F}(s_i)}(\\tilde a_i)$", "purple", 6.8)
    note(ax, (91, 28), "FMU transition\n$s_{i+1}, r_i^{\\mathrm{ext}}, r_i^{\\mathrm{int}}$", "orange", 6.4)
    note(ax, (25, 82), "Low-level replay buffer $\\mathcal{D}^{\\mathrm{lo}}$\nhourly transitions", "blue", 6.6)
    note(ax, (73, 82), "High-level replay buffer $\\mathcal{D}^{\\mathrm{hi}}$\nMS-HER + TD3 update", "green", 6.6)
    write(fig, "fig5_color_drawer_annotated")


def fig6():
    fig, ax = canvas(RAW / "fig6_cstep_base_v1.png")
    heading(ax, "Fig. 6 redraw guide - c-step goal and action interaction")
    note(ax, (12, 88), "High-level TD3: observe $s_t$, issue $g_t$ every $c$ steps", "blue", 7.7)
    note(ax, (71, 88), "Store $(s_t,g_t,\\sum r_i^{\\mathrm{ext}},s_{t+c})$\nand update $\\mathcal{D}^{\\mathrm{hi}}$", "blue", 6.8)
    note(ax, (50, 62), "Modelica-FMU energy environment\nadvanced once per hour", "gray", 7.8)
    note(ax, (15, 36), "At hour $i$: $(s_i,g_i)$", "purple", 7.2)
    note(ax, (37, 36), "Low-level TD3\n$\\tilde a_i=\\pi_{\\mathrm{lo}}(s_i,\\kappa g_i)$", "purple", 6.8)
    note(ax, (59, 36), "GiveSafe\n$a_i=\\Pi_{\\mathcal{F}(s_i)}(\\tilde a_i)$", "green", 6.8)
    note(ax, (81, 36), "FMU feedback\n$r_i^{\\mathrm{ext}}, r_i^{\\mathrm{int}}, s_{i+1}$", "blue", 6.7)
    note(ax, (50, 14), "Residual-goal update:  $g_{i+1}=h(s_i,g_i,s_{i+1})=s_i^{\\mathrm{int}}+g_i-s_{i+1}^{\\mathrm{int}}$", "purple", 6.8)
    note(ax, (50, 4.5), "Time: $t$  $\\rightarrow$  $t+1$  $\\rightarrow$  $\\cdots$  $\\rightarrow$  $t+c$", "gray", 7.2)
    write(fig, "fig6_color_drawer_annotated")


if __name__ == "__main__":
    fig1()
    fig4()
    fig5()
    fig6()
    print(f"Wrote Color Drawer redraw guides to {OUT}")
