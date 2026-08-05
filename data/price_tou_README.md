# 分时购售电价 `price_tou.csv`

**当前序列：山东省 2026 年工商业电网代理购电分时到户电价（price-taker 外生价，非现货出清结果）。**

| 列 | 含义 | 单位 |
|----|------|------|
| `time` | 仿真时间 | s，步长 3600，0…8759h |
| `buy_yuan_per_kwh` | 购电到户电度价 | 元/kWh |
| `sell_yuan_per_kwh` | 余电上网结算价 | 元/kWh |
| `band` | 分时档位 | S 尖峰 / P 高峰 / F 平 / V 低谷 / D 深谷 |

元数据：`price_tou_meta.json`。  
重建脚本：`scripts/build_price_tou_shandong.py`。

> 风光荷为 FMU `TypicalScene` **典型场景年**（温带北方算例），**未绑定山东气象站**。  
> 论文表述建议：典型场景资源边界 + 山东容积分时电价；勿写「山东实测风光荷」。

可选价区脚本（非默认）：`scripts/build_price_tou_mengxi.py`（蒙西）。  
备份：`price_tou_shandong_backup.csv` 与恢复时使用的同名备份。

---

## 数据出处

| 项 | 内容 |
|----|------|
| 时段与浮动比例 | 国网山东省电力公司 **2026 年工商业分时电价公告**；框架依据《关于进一步优化工商业分时电价政策的通知》（鲁发改价格〔2023〕914 号） |
| 整理入口 | 介子九维 smart-microgrid / 本仓库 `build_price_tou_shandong.py` |
| 代理购电与加项 | 算例参数：`proxy_price`、`capacity_comp`、`line_loss`、`sys_op`、`td_energy`、`fund` |
| 输配电度价 | 第四监管周期 · 山东 **110 kV = 0.106 元/kWh**（[介子九维输配电查询](https://www.jiezijiuwei.com/tools/transmission-tariffs) / NDRC 表） |

### 官方 PDF / 网页（核对用）

| 核对对象 | 链接 |
|----------|------|
| 鲁发改价格〔2023〕914 号 | https://www.shandong.gov.cn/module/download/downfile.jsp?classid=0&filename=b373ac6b7f11438abf4c9ff6c1bb4d0e.pdf |
| 2026 工商业分时公告（济南发改转发） | https://jndpc.jinan.gov.cn/col2191/art/2025/art_2191_4789557.html |
| 输配电价查询 | https://www.jiezijiuwei.com/tools/transmission-tariffs |
| NDRC 第四周期省级输配电价 | https://www.ndrc.gov.cn/xxgk/zcfb/tz/202607/P020260710613207914509.pdf |

---

## 购电价公式

参与浮动基数：

\[
B = 0.4 + 0.0705 + 0.01 + 0.0209 = 0.5014
\]

不参与浮动：

\[
A = 0.106 + 0.02717 = 0.13317
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

全年按 **2026 非闰年** 分月时段表展开为 8760 点。

## 售电价

\[
\lambda^{\mathrm{sell}} = 0.5\times 0.225 + 0.5\times 0.15 = 0.1875\ \mathrm{元/kWh}
\]

（算例平坦售电价。）

## 使用注意

1. 代理购电路径 price-taker；不做电网出清。  
2. 改电压等级：改 `td_energy_price` 后重跑 `build_price_tou_shandong.py`。  
3. 改价后重跑 `build_price_residual_series.py` + `train_price_bilstm.py`（若用预测价）。  
4. 在蒙西价下训过的实验需按山东价 **重评/重训** 后再入主表。  
