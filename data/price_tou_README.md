# 分时购售电价 `price_tou.csv`

**当前序列：山东省 2026 年工商业电网代理购电分时到户电价（price-taker 外生价，非现货出清结果）。**

| 列 | 含义 | 单位 |
|----|------|------|
| `time` | 仿真时间 | s，步长 3600，0…8759h |
| `buy_yuan_per_kwh` | 购电到户电度价 | 元/kWh |
| `sell_yuan_per_kwh` | 余电上网结算价 | 元/kWh |
| `band` | 分时档位 | S 尖峰 / P 高峰 / F 平 / V 低谷 / D 深谷 |

元数据见 `price_tou_meta.json`。重建脚本：`scripts/build_price_tou_shandong.py`。

## 数据出处

| 项 | 内容 |
|----|------|
| 时段与浮动比例 | 国网山东省电力公司 **2026 年工商业分时电价公告**；框架依据《关于进一步优化工商业分时电价政策的通知》（鲁发改价格〔2023〕914 号） |
| 整理入口 | 介子九维智能微电网测算工具公开编译表 `priceData.tou.provinces[shandong]`（[smart-microgrid](https://www.jiezijiuwei.com/tools/smart-microgrid)） |
| 代理购电与加项参数 | 同工具 `REFERENCE_INPUTS`（山东园区算例）：`proxy_price`、`capacity_comp`、`line_loss`、`sys_op`、`td_energy`、`fund` |
| 输配电度价 | 国家发展改革委 **第四监管周期** 省级输配电价表，算例取山东 **110 kV** 电度电价 0.106 元/kWh |

## 购电价公式

参与浮动基数：

\[
B = \lambda^{\mathrm{proxy}} + \lambda^{\mathrm{capcomp}} + \lambda^{\mathrm{lloss}} + \lambda^{\mathrm{sys}}
= 0.4 + 0.0705 + 0.01 + 0.0209 = 0.5014
\]

不参与浮动：

\[
A = \lambda^{\mathrm{td}} + \lambda^{\mathrm{fund}} = 0.106 + 0.02717 = 0.13317
\]

分时比例：尖峰 2.0、高峰 1.7、平 1.0、低谷 0.3、深谷 0.1。

\[
\lambda^{\mathrm{buy}}_{t} = B \cdot r_{\mathrm{band}(t)} + A
\]

| 档位 | 比例 | 购电价（元/kWh） |
|------|------|------------------|
| 尖峰 S | 2.0 | 1.13597 |
| 高峰 P | 1.7 | 0.98555 |
| 平段 F | 1.0 | 0.63457 |
| 低谷 V | 0.3 | 0.28359 |
| 深谷 D | 0.1 | 0.18331 |

全年按 **2026 非闰年** 分月时段表展开为 8760 点（1–2 月与 12 月、3–5 月、6 月、7–8 月、9–11 月五套 24h 模板）。

## 售电价

\[
\lambda^{\mathrm{sell}} = 0.5\times 0.225 + 0.5\times 0.15 = 0.1875\ \mathrm{元/kWh}
\]

（机制电价与上网市场均价各半；与分时档位无关。）

## 使用注意

1. **代理购电路径**，适用于电网代理购电工商业用户叙事；直接参与现货的用户电能量价由市场形成，不可直接套用本表绝对水平，但峰谷时段仍可能参照省规。
2. 电压等级改变时，请改 `td_energy_price` 后重跑 `scripts/build_price_tou_shandong.py`。
3. **不做电网出清**；本文件仅作 price-taker 外生电价。
4. 正式对外发表前建议与国网山东当年公告原文、接入电压及项目结算单再核对一次。
