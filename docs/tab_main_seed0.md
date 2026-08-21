# seasonal_v1 seed 0 — live pair template (FS-HSAC-support vs sac_param)

Source: `runs/seasonal_v1/**/train_result.json` after the results gate.
Only `valid_steps=168` rows enter `tab:main`. Truncated weeks are not ranked on cash.

**Live pair:** `fs_hsac_support` (`--method fs_hsac --support` / `FS_HSAC_NO_FEAS=1`) vs `sac_param` (`--method sac`).  
Same GiveSafe, soft_shell OFF. Full residual FS-HSAC is appendix only. No HMSD.

`docs/fs_hsac_results_gate.md` is **`gate_passed: false`**. Cells below stay blank.  
Withdrawn: winter PSO \(14.36\times10^6\) vs linprog \(10.19\times10^6\) (held-out fit; not this paper).

## tab:main (full 168 h; reserved)

| season | method | reject rate | valid_steps | CC (CNY) | Jgen |
|--------|--------|-------------|-------------|----------|------|
| winter | fs_hsac_support | — | — | — | — |
| winter | sac_param | — | — | — | — |
| transition | fs_hsac_support | — | — | — | — |
| transition | sac_param | — | — | — | — |
| summer | fs_hsac_support | — | — | — | — |
| summer | sac_param | — | — | — | — |

## tab:run (executability; reserved)

| season | method | status | hours | full_week | note |
|--------|--------|--------|-------|-----------|------|
| winter | fs_hsac_support | — | — | — |  |
| winter | sac_param | — | — | — |  |
| transition | fs_hsac_support | — | — | — |  |
| transition | sac_param | — | — | — |  |
| summer | fs_hsac_support | — | — | — |  |
| summer | sac_param | — | — | — |  |

## Forbidden claims

- Do not fill this table from older archives.
- Do not rank truncated weeks against full weeks on cash.
- Do not claim 8760 h RL safe + best economics.
- Do not report HMSD as the paper identity.
- Do not claim unconditional RL > MILP.
- Do not write superiority while the results gate is false.
