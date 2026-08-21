# 文档索引（FS-HSAC）

文档更新：2026-08-21

**现行论文提纲（唯一）：** [paper_outline_and_figures.md](paper_outline_and_figures.md)

主方法（live）：**FS-HSAC-support**（`--method fs_hsac --support` 或 `FS_HSAC_NO_FEAS=1`）。对照：固定带 Hybrid SAC（`--method sac`）。完整残余 $C_\psi$ FS-HSAC 仅附录。旧 HMSD / GHTD3 叙事已移至 [`_archive/`](_archive/README.md)，**不可当作现行口径**。结果门：`gate_passed: false`。

## 主线真源

| 项 | 路径 |
| --- | --- |
| 论文提纲 + 图表清单 | [paper_outline_and_figures.md](paper_outline_and_figures.md) |
| 贡献口径 | [ae_contributions_zh.md](ae_contributions_zh.md) |
| PAMDP / 算法形式化 | [pamdp_formalization.md](pamdp_formalization.md) |
| 消融矩阵 | [fs_hsac_ablation_matrix.md](fs_hsac_ablation_matrix.md) |
| 结果门（唯一） | [fs_hsac_results_gate.md](fs_hsac_results_gate.md) |
| 矩阵跑批状态 | [matrix_status_hybrid_sac.md](matrix_status_hybrid_sac.md) |
| 分项成本表稿 | [tab_main_seed0.md](tab_main_seed0.md) |
| 结果清单 | [paper_results_manifest.md](paper_results_manifest.md) |
| **参数证据台账** | [parameter_evidence.md](parameter_evidence.md)（profile `official-2024-ets-sd-grid-v1`） |
| 综合成本分项 | [comprehensive_cost_terms.md](comprehensive_cost_terms.md) |
| 正文 | [`Paper/main.tex`](../Paper/main.tex) |
| 训练入口 | `scripts/train_seasonal.py --method fs_hsac --support` vs `--method sac` |

## 推荐阅读顺序

1. [paper_outline_and_figures.md](paper_outline_and_figures.md) — 提纲与勿写清单  
2. [pamdp_formalization.md](pamdp_formalization.md) — \(\mathcal A(s)\) / 支撑一致 Hybrid SAC  
3. [ae_contributions_zh.md](ae_contributions_zh.md) — **一条**贡献  
4. [fs_hsac_ablation_matrix.md](fs_hsac_ablation_matrix.md) + [fs_hsac_results_gate.md](fs_hsac_results_gate.md)  
5. [cui_seasonal_min_protocol.md](cui_seasonal_min_protocol.md) — 公平周窗口（设定；论文方法已换为 FS-HSAC-support）  
6. 环境接口：`FMU输入上下限.md`、`comprehensive_cost_terms.md`、`data_dictionary.md`

## 写作辅助（保留）

| 文档 | 用途 |
| --- | --- |
| [figure_set_genre.md](figure_set_genre.md) | 图体裁 |
| [milp_baseline_notes.md](milp_baseline_notes.md) | MILP 对照口径 |
| [sensitivity_section.md](sensitivity_section.md) | 灵敏度节 |
| [carbon_price_china_ets.md](carbon_price_china_ets.md) | 碳价与官方口径 |
| [parameter_evidence.md](parameter_evidence.md) | 价格/碳参数证据等级与灵敏度 |
| [连续年SOC附录协议.md](连续年SOC附录协议.md) | 附录协议（非主文 KPI） |

## 归档

[`_archive/2026-08-hmsd/`](_archive/2026-08-hmsd/) — HMSD 身份、旧结果表、旧论文草稿索引、已废止的 `results_gate.md`。说明见 [`_archive/README.md`](_archive/README.md)。
