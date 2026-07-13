# FMU 调度输入上下限

来源：[`resources/Example/TypicalScene/PowerSystem_8760h.mo`](../resources/Example/TypicalScene/PowerSystem_8760h.mo) 中的设备实例参数，以及 [`resources/TypicalScenarios.mo`](../resources/TypicalScenarios.mo) 中的功率映射方程。

> Modelica 当前**不对** `u_tp` / `u_battery` / `u_caes` 做饱和；超出设备允许范围时由 **Python 侧校验并报错**（见 `build.md`）。

## 总表（Python 应强制）

| 输入 | 含义 | 下界 | 上界 | 备注 |
|------|------|------|------|------|
| `u_tp` | 火电负荷率（无量纲） | `1/3` ≈ `0.333` | `1.0` | 对应 `P_min/P_cap`～`P_max/P_cap`；不允许停机（`0`） |
| `u_battery` | 电池功率指令（无量纲） | `-1.0` | `1.0` | 正=充电，负=放电；额定功率归一化 |
| `u_caes` | CAES 功率指令（无量纲） | 见下方分段 | 见下方分段 | 正=充电，负=放电；含最小启机比例 |

### `u_caes` 允许集合

```text
u_caes ∈ [-1.0, -0.33]  ∪  {0}  ∪  [0.86, 1.0]
```

- 负值：压缩储能（吸收功率）
- `0`：停机/待机
- 正值：膨胀发电（释放功率）
- `|u|` 为相对额定功率的启机比例；中间开区间 `(-0.33, 0)` 与 `(0, 0.86)` **不允许**

默认初值（与模型 `start` 一致）：`u_tp=1`，`u_battery=0`，`u_caes=0`。

---

## 推导依据（PowerSystem_8760h 实例）

### 1. `u_tp` ← `thermalPower`

| 参数 | 实例值 | 说明 |
|------|--------|------|
| `P_cap` | `1.5e8` W（150 MW） | 装机容量 |
| `P_max` | `1.5e8` W | 最大出力 |
| `P_min` | `5e7` W（50 MW） | 最小稳燃出力 |

映射（[`TypicalScenarios.ThermalPower`](../resources/TypicalScenarios.mo)）：

```modelica
positivePlug.P_plan = -u_dispatch * P_cap;
positivePlug.P_act  = positivePlug.P_plan;  // 当前无饱和
```

因此负荷率与功率的关系为 `|P| = u_tp * P_cap`（发电为负）：

```text
u_tp_min = P_min / P_cap = 5e7 / 1.5e8 = 1/3 ≈ 0.333333
u_tp_max = P_max / P_cap = 1.5e8 / 1.5e8 = 1
```

内嵌调度表中亦大量出现 `0.33` 作为下限档位，与 `P_min/P_cap` 一致。

### 2. `u_battery` ← `battery`

| 参数 | 实例值 | 说明 |
|------|--------|------|
| `P_cap` | `1e8` W（100 MW） | 功率装机 |
| `SOC_min` | `0.1`（类默认） | 荷电下限 |
| `SOC_max` | `0.9`（实例覆盖） | 荷电上限 |

映射：

```modelica
PBS.P_plan = u_dispatch * P_cap;
PBS.P_act  = PBS.P_plan;
```

额定功率归一化指令箱：

```text
u_battery ∈ [-1, 1]
P = u_battery * P_cap ∈ [-100, 100] MW
```

SOC 不改变输入箱，但约束**动态可行动作**（接近空/满时不能再放/充）；Python 可用 `battery_soc` 做二次限幅。

### 3. `u_caes` ← `compressedAirEnergyStorage`

| 参数 | 实例值 | 说明 |
|------|--------|------|
| `P_cap` | `1.5e8` W（150 MW） | 功率装机 |
| 气罐 `SOC` | 默认 `[0.6, 1]` | `p/p_norm` |
| 热/冷罐 `SOC` | 默认 `[0.05, 0.95]` | 容积比 |

映射：

```modelica
PBS.P_plan = u_dispatch * P_cap;
PBS.P_act  = PBS.P_plan;
```

功率归一化幅值箱为 `[-1, 1]`；工程上另限制最小启机比例（与历史调度脚本、内嵌 CAES 表一致）：

```text
放电（膨胀）：u_caes ∈ [-1, -0.33]
待机：         u_caes = 0
充电（压缩）：u_caes ∈ [0.86, 1]
```

罐 SOC / 压力温度由 FMU 输出；接近边界时由 Python 做动态限幅。

---

## 校验伪代码（建议）

```python
TP_LO, TP_HI = 1.0 / 3.0, 1.0
BAT_LO, BAT_HI = -1.0, 1.0

def check_u_tp(u: float) -> None:
    if not (TP_LO <= u <= TP_HI):
        raise ValueError(f"u_tp={u} out of [{TP_LO}, {TP_HI}]")

def check_u_battery(u: float) -> None:
    if not (BAT_LO <= u <= BAT_HI):
        raise ValueError(f"u_battery={u} out of [{BAT_LO}, {BAT_HI}]")

def check_u_caes(u: float, eps: float = 1e-9) -> None:
    ok = (
        abs(u) <= eps
        or (-1.0 <= u <= -0.33)
        or (0.86 <= u <= 1.0)
    )
    if not ok:
        raise ValueError(
            f"u_caes={u} not in [-1,-0.33] U {{0}} U [0.86,1]"
        )
```

---

## 与旧文档的差异

| 项 | 旧 `data_dictionary.md` | 本文（按当前 resources） |
|----|-------------------------|--------------------------|
| `u_tp` 下界 | `0.34` | `1/3`（`P_min/P_cap`，与表中 `0.33` 一致） |
| `u_battery` | `[-1, 1]` | 不变 |
| `u_caes` | 分段集合 | 不变（工程启机带 + 额定归一化） |
