# Parameter evidence ledger (official + literature)

文档更新：2026-08-30 20:20 (+08:00)  
**Active profile:** `official-2024-ets-sd-grid-v1`（见 `src/config/reward_config.yaml`）  
**Legacy profile (archived runs):** `legacy-2022-grid-factor/proxy-benchmark`（`runs/**/config/reward_config.yaml` 快照勿改）

主排名仍是完整 168 h 综合成本 \(CC=-J^{\mathrm{gen}}\)。本表只回答“每个价格参数凭什么取这个数”。

证据等级：

| 等级 | 含义 |
|------|------|
| **O** Official observation / statute | 政府/监管机构公开数值 |
| **M** Mechanism official | 官方只支持机制/结构，绝对值需构造 |
| **L** Literature case | 同行评议算例参数（量级交叉验证） |
| **S** Scenario / extrapolation | 本研究情景或外推；须灵敏度 |

---

## 1. Carbon settlement

| 字段 | 主值 | 等级 | 出处与边界 |
|------|------|------|------------|
| `carbon.price_cny_per_t` | **97.49** | O | 生态环境部：2024 年全国碳市场年底综合收盘价 97.49 CNY/t；全年 69–106，峰值 105.65。旧主值 80 为 ICAP/区间中位代理，**非**最新官方单值。灵敏度：`{69.30, 80, 86.4, 97.49, 105.65}`（86.4≈GHTD3 \(0.012\) USD/kg×7.2） |
| `carbon.mode` | `intensity_benchmark` | M+S | 对齐全国 ETS 基准法配额思路；周 episode 末结算是模型简化，不是监管结算日 |
| `carbon.beta_t_per_mwh` | **0.8049** | O | 150 MW 机组属「300 MW 等级及以下常规燃煤」；《2023、2024 年度全国碳排放权交易发电行业配额总量和分配方案》2024 发电基准值 0.8049 tCO₂/MWh。旧 0.82 为相对 η_th 的代理 |
| `carbon.eta_grid_t_per_mwh` | **0.6191** | O | 生态环境部/国家统计局《2023 年电力二氧化碳排放因子》：山东位置型平均因子 0.6191 kgCO₂/kWh = 0.6191 t/MWh。旧 0.5703 为 2022 年全国电网平均（报告管理通知口径） |
| `carbon.eta_thermal_t_per_mwh` | **0.85** | S | 机组直接排放情景代理（非配额基准）。FMU/Modelica 侧无独立燃料–CO₂ 出口可反算 η_th，故保留 Python 情景值。官方 2023 平衡值≈0.8155、2024 基准 0.8049（同机组类别）作灵敏度。**禁止**把 η_th 与 β 混称为同一官方量 |

公式（代码真源 `RewardCalculator`）：

- 配额：\(A+=\beta E_{\mathrm{th}}\)，排放：\(E+=\eta_{\mathrm{th}}E_{\mathrm{th}}\)；期末 \(C^{\mathrm{CO_2,th}}=-\pi Q\)，\(Q=A-E\)。
- 购电：逐步 \(C^{\mathrm{CO_2,g}}=\pi\eta_g E_{\mathrm{buy}}\)（`grid_in_quota: false`）。

官方 PDF（核验）：

- 全国碳市场发展报告（2025）：`https://www.mee.gov.cn/ywgz/ydqhbh/wsqtkz/202509/W020250927515316322073.pdf`
- 2024 配额交易清缴新闻：`https://www.mee.gov.cn/ywgz/ydqhbh/syqhbh/202501/t20250105_1099975.shtml`
- 配额方案：`https://www.mee.gov.cn/xxgk2018/xxgk/xxgk03/202410/W020241021392230468687.pdf`
- 2023 电力因子：`https://www.mee.gov.cn/xxgk2018/xxgk/xxgk01/202512/W020251231726284332528.pdf`

---

## 2. TOU purchase / sell

文档更新：2026-08-25 21:00 (+08:00)

| 字段 | 主值 | 等级 | 出处与边界 |
|------|------|------|------------|
| 分时时段与浮动比例 | 五档 S/P/F/V/D，分五组月 | O/M | 鲁发改价格〔2023〕914 号 + 国网山东 2026 工商业分时公告（时段沿用 2025） |
| 到户购电绝对值 | **按月** 110 kV 两部制电度五档 | O（2026-01…08）/ S（09–12 顺延 08） | 国网山东每月《代理购电工商业用户电价表》。台账 `data/price_tou_monthly_official.csv`。**禁止**再用全年五档常数 1.136/0.986/0.635/0.284/0.183 冒充官方到户 |
| `td_energy` | 0.1191（1–7 月）/ **0.106**（8 月起） | O | 第三监管周期 110 kV 两部制电量电价至 2026-07-31；鲁发改价格〔2026〕556 号自 2026-08-01 为 0.106 |
| 售电 0.1875 | CNY/kWh | S | 平坦出口算例；非官方月度到户 |

论文口径：**月度代理购电到户分时 × 官方分月时段**，price-taker，非 ISO 出清。2026-01…08 已从官方表填入；09–12 顺延 8 月表（S），待后续月度公告替换。

---

## 3. Curtailment / unserved

| 字段 | 主值 | 等级 | 出处与边界 |
|------|------|------|------------|
| `nu_curt_cny_per_mwh` | **300** | L | GHTD3 Table 2：\(\nu_{\mathrm{res}}=0.041\) USD/kWh ≈ **295** CNY/MWh（×7.2）；CAES 消纳文献常见 350。目标函数惩罚，非监管弃电价 |
| `nu_uns_cny_per_mwh` | **1000** | S | 高于购电尖峰量级的保守缺供惩罚；非全国统一 VOLL（文献综述量级见 Schröder & Kuckshinrichs 2015）。灵敏度：×电价 4–5 倍赔偿口径 / 更高 VOLL 档 |

---

## 4. Battery degradation

| 字段 | 主值 | 等级 | 出处 |
|------|------|------|------|
| \(\psi(\delta)=a_0\delta^{2.03}\) | — | L | Cui OCTD3 / GHTD3 |
| `capex_cny_per_kwh` 1000 | L | Xu 2022 LCOS：LFP 能量成本约 800–1300 CNY/kWh |
| `n_cycles` 5000, `dod_eq` 0.8 | L | 同族 LFP 寿命量级 |

---

## 5. CAES startup

| 字段 | 主值 | 等级 | 出处与边界 |
|------|------|------|------------|
| `c_su_usd_ref` 3.42 | L | Cui 2024 Table 2：800 kW 算例经验系数 |
| 线性缩放到 150 MW → ≈4617 CNY | **S** | **外推**，原文未授权容量线性放大。灵敏度：不缩放 / 线性 / \(\sqrt{P}\) 区间 |

---

## 6. Grid contract (Story A)

| 字段 | 主值 | 等级 | 出处与边界 |
|------|------|------|------------|
| `p_lim_w` ±200 MW | S | 容量情景；FMU 硬限仍 ±500 MW |
| `nu_cny_per_mwh` 600 | S | 情景服务费；偏差电量常按交易价倍数结算（机制文献），**非**固定监管 600。灵敏度：`{0, 0.5, 1, 2}×` |

---

## 7. DRL trainer knobs (Cui 2024 Table 4)

Cui *Appl. Energy* 374 (2024) 123950 Table 4 is **OCTD3**. Copy only same-hour DRL numbers. Do **not** copy options, \(\kappa\), \(D=2\), \(\sigma^2\), or \(w=0.2\) PFI/CCI.

| 字段 | Cui Table 4 | 本代码 | 说明 |
|------|-------------|--------|------|
| \(\varepsilon_{\max},\varepsilon_{\min}\) | 1.0, 0.05 | 同 | `hybrid_common/explore.py` |
| \(\Delta\varepsilon\) | \(6\times10^{-6}\) @ \(2\times10^5\) step | 按 horizon 比例缩放 | 周协议 \(8.4\times10^5\) step 时约 \(1.43\times10^{-6}\)，\(\varepsilon\) 仍在 \(\approx79\%\) 处落到 0.05 |
| lr | \(10^{-4}\) | \(10^{-4}\) | actor/critic/\(\alpha\) |
| \(\tau\) | 0.005 | 0.005 | |
| batch | 64 | 64 | |
| \(\gamma\) | 0.99 | 0.99 | |
| replay | \(10^4\) | \(4.2\times10^4\) | \(\propto\) 步数 |
| \(R^F\) mix | Eq. (35) \(0.5\) | 0.5 | 只进 \(r^{\mathrm{ext}}\) |
| \(\theta_{\mathrm{thr}}\) | “历史数据”，无数字 | 50 MW | 本厂 100 MW 电池一半；**S** |
| SAC \(\alpha\) clip | — | \([0.05,2]\) | Haarnoja；不是 Cui \(\kappa=1\) |

---

## 8. How reference papers organise parameters (adopt / reject)

**Adopt (GHTD3 / OCTD3):**

- 综合成本公式分项写清；参数表列主值；正文给来源；价格类灵敏度。
- 机制消融（设备组）与算法消融分开；成本雷达/分项，不把协同收益塞进算法名次。
- 弃电/碳价用文献量级交叉验证。

**Reject:**

- 不采用 OCTD3 的 PFI/CCI/SRSI 加权总评分替代 \(CC\)。
- 不把海拉尔算例 TOU、0.832 kg/kWh、限额服务费写成山东官方值。
- 不把 `c_su=3.42 USD` 说成法规启停价或未声明的容量外推。

---

## 9. Code ↔ paper field map

| 论文符号 | YAML / kpi |
|----------|------------|
| \(\pi_{\mathrm{CO_2}}\) | `carbon.price_cny_per_t` |
| \(\beta\) | `carbon.beta_t_per_mwh` |
| \(\eta_{\mathrm{th}},\eta_{\mathrm{grid}}\) | `eta_thermal_*`, `eta_grid_*` |
| \(\nu_{\mathrm{curt}},\nu_{\mathrm{uns}}\) | `curtailment.nu_*` |
| \(C^{\mathrm{su}}\) | `caes_startup_*` |
| \(C^{\mathrm{grid}}\) | `grid_contract.*` |
| profile id | `parameter_profile_id` in `reward_config.yaml` / `train_result.json` |
