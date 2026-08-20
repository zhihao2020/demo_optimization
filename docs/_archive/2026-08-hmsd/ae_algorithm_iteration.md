<!-- ARCHIVED: superseded by FS-HSAC; see docs/paper_outline_and_figures.md -->

# 算法迭代协议（不能保证接收）

文档更新：2026-08-17 12:30 (+08:00)

Applied Energy **无法保证接收**。禁止为了「看起来更新」而堆模块。

## 一句话算法

**Story A 主方法**：二维 HMSD（`ghtd3_config.yaml`）+ \(J\) 含 ±200 MW 联络线合同罚。  
**锁死压空对照**：`python scripts/train_seasonal.py ... --lock-caes`。  
**HMSD-aligned**：4 维定额，夏天已证伪，不当主方法。  
**HMSD-W / quota**：仅电池定额（消融）。

轻罚-only 的 `runs/wear/*_s0`：冬 \(R=115.9\) 过关，夏 \(J\) 略升但电池吞吐 5581>4328，定额没卡住。保留作对照，不写进主方法。执行裁剪后重跑 `runs/quota/*_s0`。

## 证伪（跑完再写进论文）

| 轮 | 变体 | 过关 | 失败就停 |
|----|------|------|----------|
| 0 | 二维 HMSD | 冬修闲置 | — |
| 1 | Cui 式（低层追子目标） | 对照表 | — |
| 2 | 轻罚 W | 夏吞吐应降 | **未过**（吞吐升）→ 不当主方法 |
| 3 | 执行裁剪 quota W | 夏吞吐降或 \(J\) 升，冬 \(R\gtrsim 90\) | 冬崩且夏不涨 → 不把 W 写进主方法 |
| 4 | HMSD-B | 夏火电接近 TD3（约 9000）且 \(J\) 升 | 冬崩 → 只保留 W 或回到二维 |

禁止：市场先验、预热、再换演员来凑创新。

## 服务器

- Cui 式：`runs/cui_style/{winter,transition,summer}_s0`  
- 轻罚 W（对照）：`runs/wear/{winter,summer}_s0`  
- 执行裁剪 W（连续 \(u_{\mathrm{caes}}\) 消融）：`runs/quota/{winter,summer}_s0`  
- 对齐（对照）：`runs/aligned/{winter,summer}_s0`（`ghtd3_aligned.yaml`，不当主方法）  
- Story A 锁死压空：`runs/seasonal/{season}/{method}_lockcaes_s0`  
