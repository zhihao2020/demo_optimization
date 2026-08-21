# FS-HSAC results gate

```yaml
gate_passed: false
algorithm: fs_hsac_v2
live_method: fs_hsac_support
contrast: sac_param
blocker: "seasonal seed-0 fs_hsac_support vs sac_param not yet completed; unit gates passed"
```

This is the **only** results gate for the current paper. The former hybrid-SAC `results_gate.md` is archived under `docs/_archive/2026-08-hmsd/`.

The live pair (when it exists): seasonal seed-0 `fs_hsac_support` with `FS_HSAC_NO_FEAS=1` / `--support` versus `sac_param` (`--method sac`, `parameterized_caes=True`). Same GiveSafe, soft_shell OFF. Metrics later: reject rate, `valid_steps=168`, comprehensive cost. Full residual-feasibility FS-HSAC is appendix only.

Pass when (bookkeeping only; **do not** treat this list as a superiority claim in the manuscript):

1. `fs_hsac_support_s0` and `sac_param_s0` exist for the reported seasons.
2. Economic rows require `valid_steps=168`.
3. Reject rate and comprehensive cost are taken from the same eval dump.

Until then: do not claim superiority in `Paper/main.tex`. Do not fill results tables with archive cash pairs (including withdrawn winter PSO vs linprog).
