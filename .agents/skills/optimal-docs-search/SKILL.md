---
name: optimal-docs-search
description: >-
  Search optimal_demo living docs via local qmd before answering architecture,
  training stack, CAES/action space, GiveSafe, seasonal protocol, or config
  questions. Also use when aligning docs after a behavioral code change
  (find related pages with qmd, update, stamp, reindex). Code edits still use
  Grep/Read. Triggers: HMSD, GHTD3, seasonal fair, u_caes, GiveSafe,
  「文档怎么说」, docs-code alignment.
---

# optimal_demo Docs Search (qmd)

## When to use

1. **Answer from docs** — before inventing architecture / training / safety answers, **search first**.
2. **Align docs after behavioral code change** — before editing living docs or proposing commit, **qmd-find related pages**, then update + stamp + reindex.

Triggers: HMSD/GHTD3、CAES/`u_caes`、GiveSafe、seasonal train/eval、配置路径、hybrid 基线边界、「文档怎么说」、行为变更后的文档对齐。

**Not this skill:** code navigation (use Grep/Read on `src/`); paper LaTeX body (`Paper/main.tex`) unless user asks paper-docs alignment.

## Tool split (mandatory)

| Work | Tool | Rule |
| --- | --- | --- |
| Change / navigate **code** | Grep / Read / Glob | Do **not** require qmd for code |
| Decide which **living docs** to touch; write/align docs | **qmd** `search` / `query` | **Must** use qmd before changing docs |
| After doc edits | `scripts/qmd/reindex.ps1` | Keep index current |

## Docs ↔ code alignment timing

```text
Code change (semantic / contract / behavior)   ← Grep / Read
  → qmd search/query related living docs / README
  → edit those docs in the same work unit (if needed)
  → stamp 文档更新：YYYY-MM-DD HH:MM (+08:00)
  → powershell -File scripts\qmd\reindex.ps1
  → git add code + docs in the same commit
```

Docs are part of the change set and must already match code **before** `git add`.

### When docs must change

| Change | Update |
| --- | --- |
| Action space / CAES / validator / GiveSafe | `README.md`, `docs/FMU*`, algo living |
| Train algorithm / goals / buffers / networks | `docs/GHTD3分层实现说明.md`, `docs/principle_innovation_MIF_HRL.md` |
| Seasonal / fair eval / train entry | `docs/cui_seasonal_min_protocol.md`, `README.md` |
| Config paths / legacy archive | `README.md`, `docs/README.md` |
| New experiment numbers | snapshot results + manifest |
| Internal-only / no external behavior | **No** living-doc edit, **no** stamp churn |

## Prerequisites

```powershell
npm install -g @tobilu/qmd
powershell -File scripts\qmd\setup_collections.ps1
# After doc edits:
powershell -File scripts\qmd\reindex.ps1
# Optional vectors (heavy first download):
powershell -File scripts\qmd\reindex.ps1 -Embed
```

Always run qmd from the **repo root** so the project-local `.qmd/` index is used.

## Collections

| Collection | Scope |
| --- | --- |
| `readme` | Root `README.md` |
| `docs-algo` | Algorithm / principles / seasonal protocol (English-prefixed globs) |
| `docs-env` | FMU / reward / data / Modelica |
| `docs-protocol` | Protocols & readiness |
| `docs-results` | Snapshots & paper drafts |
| `docs-all` | Full `docs/**/*.md` (Chinese filenames fallback) |

Excluded by design: `runs/`, `logs/`, `reference_papers/`, `Paper/*.synctex.gz`, **source code** (use Grep).

## Mandatory flow — answering from docs

1. Pick collection(s) (or omit `-c` / use `docs-all` for Chinese titles).
2. Keyword first (fast):

```powershell
qmd search "HMSD GHTD3" -c docs-algo --format json -n 8
```

3. If BM25 is thin and embeddings exist, hybrid:

```powershell
qmd query "fair seasonal comparison" -c docs-protocol --format json -n 8 --no-rerank
```

4. Open hits with Read / `qmd get … --full`.
5. Only then answer. Prefer code truth if docs and code disagree — then fix docs.

## Mandatory flow — aligning docs after code change

1. Finish code understanding with Grep/Read (no qmd required for code).
2. `qmd search` / `qmd query` to find every living page that describes the changed behavior.
3. Edit those pages (and `docs/README.md` mainline table if boundaries change).
4. Stamp `文档更新：YYYY-MM-DD HH:MM (+08:00)` on substantively edited files only.
5. Run `powershell -File scripts\qmd\reindex.ps1`.
6. Include code + docs in the **same** commit when the user asks to commit.

## Suggested intents → collections

| Intent | Prefer | Example `qmd search` |
| --- | --- | --- |
| Seasonal fair / train weeks | `docs-protocol`, `docs-algo` | `seasonal train_seasonal her_mix` |
| CAES continuous / u_caes | `readme`, `docs-env` | `u_caes continuous` |
| Hierarchy / subgoal / HER | `docs-algo` | `subgoal_interval her_mix goal` |
| GiveSafe / Oracle | `docs-algo`, `docs-all` | `GiveSafe FeasibilityOracle` |
| Config mainline vs legacy | `readme`, `docs-all` | `ghtd3_config legacy` |
| FMU bounds | `docs-env` | `FMU u_tp u_battery` |

### BM25 tips

- Prefer short queries; long AND-like bags often return `[]`.
- Chinese-heavy pages: search English identifiers (`GHTD3`, `GiveSafe`, `caes_u`, `train_seasonal`) or use `-c docs-all`.
- After `qmd embed`, `qmd query "..."` improves semantic recall (first run downloads GGUF).

## Current mainline (do not contradict)

- Physical action: `u_tp`, `u_battery`, `u_caes` — **no** policy dims `caes_mode`/`caes_magnitude`.
- HMSD: `execution_mode: goal_conditioned`, `goal_dim: 2`, `low_reward: ext`, no hybrid residual teacher.
- Baselines: Hybrid-TD3 / Hybrid-SAC only.
- Deleted / not mainline: Hybrid-PPO, `residual_mle`, `hybrid_anchor`, Decoder, LTAR, STFR/TRAP, TEA residual as runnable mainline.
- Do **not** resurrect deleted docs: `LTAR_formulation.md`, `STFR_TRAP_formulation.md`, `GHTD3算法改进说明.md`, `Safe_Market_GHTD3_principles.md`, old `论文对照_*` ares tables.

Policy hub: [`docs/README.md`](../../docs/README.md).
