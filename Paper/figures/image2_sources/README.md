# Image-2 source-asset manifest

The PDF figures use vector labels, formulae, and arrows for journal-size
legibility. Image-2 inserts must contain no text and can be placed in
`final/` when generation access is available.

| ID | Intended use | Prompt summary | Status |
| --- | --- | --- | --- |
| `renewables_cluster` | Fig. 1 supply | Isolated wind farm and PV array; clean technical isometric style; white; no text or arrows. | Generated and embedded |
| `caes_storage_module` | Fig. 1/5 | CAES compressor-expander, vessel, thermal tank, battery; white; no text or arrows. | Generated and embedded |
| `digital_twin_terminal` | Fig. 1/5 | Monitor with unreadable abstract energy curves plus twin cube; white; no readable UI text. | Generated and embedded |
| `feasibility_emblem` | Fig. 4/5 | Safety shield around actuator and state gauge; white; no text or arrows. | Generated and embedded |
| `fig1_system_base_v1` | Fig. 1 redraw candidate | Complete four-layer integrated-energy-system scene with physical power/thermal/control routing; no text. | Generated; candidate for vector annotation |
| `fig4_reward_safety_base_v2` | Fig. 4 redraw candidate | Three reward sources converging to aggregation, policy, safe action projection, and FMU plant; no text. | Generated; candidate for vector annotation |
| `fig5_hierarchical_rl_base_v1` | Fig. 5 redraw candidate | Dense two-level RL, replay-memory, energy-environment and constraint topology; no text. | Generated; candidate for vector annotation |
| `fig6_cstep_base_v1` | Fig. 6 redraw candidate | Hourly low-level loop nested in a c-step high-level interaction timeline; no text. | Generated; candidate for vector annotation |

## Attempt record

- 2026-08-06: the built-in Image-2 path was attempted twice and failed before
  output with an image-service network error.
- 2026-08-06: the CLI `gpt-image-2` fallback is unavailable because this
  environment has no `OPENAI_API_KEY` configured.
- 2026-08-06: a subsequent built-in Image-2 attempt succeeded. Four original
  outputs are retained in `raw/`; chroma-key-extracted alpha PNGs in `final/`
  are used by `scripts/build_image2_concept_figures.py`.
- 2026-08-06: four complete, text-free base scenes were generated with the
  built-in Image-2 model and retained under `raw/`. They are source candidates,
  not publication figures: all notation, labels, formulae, and checked arrows
  still have to be redrawn as editable vector overlays.
