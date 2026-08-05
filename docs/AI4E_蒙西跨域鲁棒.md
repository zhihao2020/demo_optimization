# AI4E 蒙西跨域鲁棒实验（域 B）

## 设定

| 项目 | 域 A（主文） | 域 B（本文件） |
|------|--------------|----------------|
| 模型 | Modelica/FMU 多能厂站 | **无 FMU**；小时储能窗口调度 |
| 边界 | TypicalScene 典型年 | AI4E 蒙西 2025 系统级负荷/风/光 |
| 电价 | 山东分时容积价 | 蒙西节点 **实时价** |
| 分辨率 | 1 h | 15 min→1 h |
| 划分 | 周 episode | 2025-01..09 训 / 10..12 测 |

## 证明什么

- **证明**：`预报/特征 → 可行充放电窗口决策` 在真实蒙西系统边界与实时价上仍可获得正收益，并相对朴素滞后规则更优。
- **不证明**：同一 CAES-FMU 已换成蒙西气象年（主物理仍为 FMU 典型场景）。
- **与主方法关系**：主文 Safe Market-GHTD3 在域 A；域 B 验证同一 **「信息→约束决策」** 范式的跨域有效性（动作空间适配为储能块）。

## 测试集结果

| Method | 日均收益 | 中位收益 | 标准差 | 有操作日比例 |
|--------|---------:|---------:|-------:|-------------:|
| idle | 0.0000 | 0.0000 | 0.0000 | 0.00 |
| lag24_rule | 0.8654 | 0.9796 | 1.8048 | 1.00 |
| feature_gbdt_rule | 1.8426 | 1.9744 | 1.3341 | 1.00 |
| oracle_rule | 2.6036 | 2.3773 | 1.4385 | 1.00 |

- feature_gbdt_rule vs lag24: **+0.9773** 日均收益
- feature_gbdt_rule vs idle: **+1.8426**
- 相对 oracle 上界捕获率: **70.8%**

数据与指标 JSON：`D:\Code\0622\optimal_demo\runs\ai4e_domain_b_robustness\domain_b_results.json`

## 复现

```powershell
$env:PYTHONPATH = "src"
python scripts/prepare_ai4e_mengxi_scenario.py
python scripts/eval_ai4e_robustness.py
```

## 致谢（论文可用）

蒙西地区系统边界条件与节点实时电价数据来自第四届世界科学智能大赛（AI+能源电力）赛题；
赛题气象 NWP 由中科天机气象科技有限公司提供。作者对赛事主办方与数据提供方表示感谢。
作者为参赛选手，数据用于学术研究并按赛题要求致谢。

English: *Mengxi system-level boundaries and real-time nodal prices are from the AI4E track of the World Scientific Intelligence Competition; NWP fields were provided by TJ Weather. The authors participated in the competition and gratefully acknowledge the organizers and data providers.*
