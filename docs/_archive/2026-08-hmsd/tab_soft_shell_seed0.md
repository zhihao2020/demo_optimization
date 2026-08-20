<!-- ARCHIVED: superseded by FS-HSAC; see docs/paper_outline_and_figures.md -->

# Soft-shell held-out eval (seed 0)

Eval-only shell on frozen seasonal_v1 weights. Not the hard-protocol main table.

| season | method | status | valid_steps | soft_shell_count | episode_reward |
|--------|--------|--------|-------------|------------------|----------------|
| winter | hmsd | completed | 168 | 2 | 75.7943661365512 |
| winter | td3 | completed | 168 | 0 | 89.82345141737139 |
| winter | sac | completed | 168 | 0 | 83.02941486121348 |
| transition | hmsd | completed | 168 | 0 | 43.43195925610688 |
| transition | td3 | completed | 15 | 0 | 0.9224570381608793 |
| transition | sac | completed | 50 | 0 | 2.3734995971315698 |
| summer | hmsd | completed | 168 | 2 | 49.63347881765635 |
| summer | td3 | completed | 76 | 0 | 11.359749696126274 |
| summer | sac | skipped_missing_checkpoint | None | None | None |
