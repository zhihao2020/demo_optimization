已按计划改完 Modelica，摘要如下。

**[`PowerSystem_8760h.mo`](resources/Example/TypicalScene/PowerSystem_8760h.mo)**
- 删除：`OPT_goal`、`P_res`、全部 `*_penalty`
- 新增物理输出：`p_curtailment` / `p_unserved`、CAES SOC、设备功率、风光可用/实际、`p_load_actual`、CAES 压力与温度
- 补充职责边界、单位与符号约定注释

**[`TypicalScenarios.mo`](resources/TypicalScenarios.mo)**
- ThermalPower / Battery / CAES / Bus 中经济罚函数全部置 `0`，并注明迁出原因
- 设备相关 `e^` 罚公式已清除

Python 未改；导出新 FMU 后再对齐 `src/fmu` 输出名即可。