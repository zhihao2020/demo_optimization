# Results bookkeeping (PC-HybridTD3)

文档更新：2026-08-30 22:40 (+08:00)

```yaml
gate_passed: false
algorithm: pc_hybrid_td3
live_method: td3
contrast:
  - td3_proj
  - td3_static
  - milp
  - rule
blocker: "Stage D TEST-week campaign not completed; do not fill Paper/main.tex tables"
```

This file is **bookkeeping only**. It is **not** the paper identity and not a superiority claim. Archived FS-HSAC / `seasonal_v1` cash must not enter `Paper/main.tex`.

Pass when (still not a manuscript ranking claim):

1. `pc_hybrid_td3_s{0,1,2}` and matching projection / static-support / MILP / rule rows exist on TEST weeks.
2. Economic rows require `valid_steps=168`.
3. Reject rate, \(E_{\mathrm{terminal}}\) and comprehensive cost come from the same eval dump.

Until then: empty cells stay empty.
