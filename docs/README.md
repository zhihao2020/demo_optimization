# 文档索引（PC-HybridTD3）

文档更新：2026-08-31 13:10 (+08:00)

**现行论文提纲（唯一）：** [paper_outline_and_figures.md](paper_outline_and_figures.md)

主方法（live）：**PC-HybridTD3**（`--method td3`，联合 \(\mathcal A_f(s)\)）。研究对象是多能源安全经济协同调度。对照：投影连续 TD3、滚动 MILP、price-aware rule。分量支撑 hybrid TD3 为支撑消融。FS-HSAC / HMSD 叙事不作为现行论文身份。结果表在 Stage D 之前保持空。

## 主线真源

| 项 | 路径 |
| --- | --- |
| 论文提纲 + 图表清单 | [paper_outline_and_figures.md](paper_outline_and_figures.md) |
| 贡献口径 | [ae_contributions_zh.md](ae_contributions_zh.md) |
| PAMDP / 算法形式化 | [pamdp_formalization.md](pamdp_formalization.md) |
| 消融矩阵 | [fs_hsac_ablation_matrix.md](fs_hsac_ablation_matrix.md)（现为 PC-HybridTD3 两组消融） |
| 结果门（书账） | [fs_hsac_results_gate.md](fs_hsac_results_gate.md)（**不是**论文 gate；表空到 Stage D） |
| 矩阵跑批状态 | [matrix_status_hybrid_sac.md](matrix_status_hybrid_sac.md) |
| 分项成本表稿 | [tab_main_seed0.md](tab_main_seed0.md) |
| 结果清单 | [paper_results_manifest.md](paper_results_manifest.md) |
| **参数证据台账** | [parameter_evidence.md](parameter_evidence.md)（profile `official-2024-ets-sd-grid-v1`） |
| 综合成本分项 | [comprehensive_cost_terms.md](comprehensive_cost_terms.md) |
| 检查清单工程报告 | [pc_hybrid_td3_check_report.md](pc_hybrid_td3_check_report.md) |
| 正文 | [`Paper/main.tex`](../Paper/main.tex) |
| 训练入口 | `scripts/train_seasonal.py --method td3 --season all`（投影：`--ablation projection`；静态支撑：`--ablation static-support`） |

## 推荐阅读顺序

1. [paper_outline_and_figures.md](paper_outline_and_figures.md) — 提纲与勿写清单  
2. [pamdp_formalization.md](pamdp_formalization.md) — \(\mathcal A_f(s)\) / PC-HybridTD3  
3. [ae_contributions_zh.md](ae_contributions_zh.md) — **一条**贡献  
4. [fs_hsac_ablation_matrix.md](fs_hsac_ablation_matrix.md) + [fs_hsac_results_gate.md](fs_hsac_results_gate.md)  
5. [cui_seasonal_min_protocol.md](cui_seasonal_min_protocol.md) — 36/8/8 周拆分  
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
