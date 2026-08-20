# FS-HSAC results gate

```yaml
gate_passed: false
algorithm: fs_hsac_v2
blocker: "full seasonal runs not yet completed; unit gates passed"
```

This is the **only** results gate for the current paper. The former hybrid-SAC `results_gate.md` is archived under `docs/_archive/2026-08-hmsd/`.

Pass when:

1. `fs_hsac_s0` completes 168 h on winter / transition / summer.
2. CC better than projected SAC and PSO on those full weeks.
3. Ablation shows dynamic support and (if enabled) classifier each help on at least one reported metric (reject rate or cost).

Until then: do not claim superiority in `Paper/main.tex`.
