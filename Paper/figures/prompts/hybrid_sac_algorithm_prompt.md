# Hybrid-action SAC algorithm figure

Use `scripts/plot_paper_v2_figures.py` as the source of truth. Do not redraw HMSD / c-step hierarchy.

The figure must show: FMU state → hybrid SAC (thermal, battery, CAES mode+mag) → legal map with F(s) mask → GiveSafe (adopted, no fallback) → FMU hour → r^ext from J^gen. No high-level goal box.
