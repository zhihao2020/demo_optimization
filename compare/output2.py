import numpy as np

# 最终结果148025544.08分
#
# =========================================================================
# 【优化变量含义与格式说明】
# 1. caes_seq (压缩空气储能系统 compressedAirEnergyStorage)
#    - 格式：一维列表(1D List)，长度默认 8760，每个元素对应1小时的指令。
#    - 范围：[-1, -0.33] U [0.86, 1] (或0)。负数代表压缩空气（储纳多余电量），正数代表膨胀（释放由于用电高峰），绝对值代表启动比例。
#
# 2. battery_seq (锂电池系统 battery)
#    - 格式：一维列表(1D List)，长度默认 8760。
#    - 范围：[-1, 1]。负数表示充电 (-1 为最大充电功率)，正数表示放电 (1 为最大放电功率)。
#
# 3. tp_seq (火电系统 thermalPower)
#    - 格式：一维列表(1D List)，长度默认 8760。
#    - 范围：[0.33, 1]。由于火电锅炉有最小稳定燃烧负荷，必须始终维持在至少 33% 功率以上。
#
#
# 【如何保存与调用这三个序列的说明指南】
# 
# 1. 保存为文件 (例如保存为 CSV 格式):
#    如果你需要将生成的策略提取出来供其他软件使用，可以在 main() 或外部脚本中这样处理：
#    import pandas as pd
#    caes, battery, tp = generate_energy_strategy(steps=8760)
#    df = pd.DataFrame({
#        "time": range(8760),  # 时间轴（有些仿真软件需要第一列是时间，单位视情况设为小时或秒）
#        "caes_seq": caes,
#        "battery_seq": battery,
#        "tp_seq": tp
#    })
#    df.to_csv("energy_strategy.csv", index=False)
#
# 2. 放入仿真软件 (如 同元 MWorks / FMU 模型) 运行:
#    - 方法 A (作为数据源模块直接导入软件):
#      大多数物理仿真平台（如 MWorks, Dymola）都支持读取外部文件作为时间序列输入。
#      你可以使用类似于 `Modelica.Blocks.Sources.CombiTimeTable` 的组件，把上面生
#      成的 "energy_strategy.csv" 设为文件路径即可让模型在运行时读取各步的控制指令。
#    
#    - 方法 B (在基于 Python 的 FMI/FMU 协同仿真中调用):
#      如果你是用 Python 脚本驱动 FMU 模型，就在仿真总循坏里查表赋值：
#      import pandas as pd
#      strategy_data = pd.read_csv("energy_strategy.csv")
#      for step in range(8760):
#          # 获取当前步的控制信号
#          caes_val = strategy_data.loc[step, "caes_seq"]
#          battery_val = strategy_data.loc[step, "battery_seq"]
#          tp_val = strategy_data.loc[step, "tp_seq"]
#          
#          # 将控制信号输入到仿真模型里 (伪代码)
#          # fmu_model.setReal([caes_ref, batt_ref, tp_ref], [caes_val, battery_val, tp_val])
#          # fmu_model.doStep(currentCommunicationPoint=step*3600, communicationStepSize=3600)
# =========================================================================

# EVOLVE-BLOCK-START
def generate_energy_strategy(steps=8760):
    # Initialize arrays
    tp_seq = np.ones(steps) * 1.0
    battery_seq = np.zeros(steps)
    caes_seq = np.zeros(steps)
    
    # Battery parameters
    battery_capacity_j = 500 * 1000 * 3600  # 1.8e9 J
    battery_power_w = 100e6  # 100 MW
    charge_factor = (0.85 * battery_power_w * 3600) / battery_capacity_j
    discharge_factor = (1/0.85 * battery_power_w * 3600) / battery_capacity_j
    
    # CAES parameters and state tracking
    gas = 0.85
    hot = 0.5
    cold = 0.5
    caes_state = 0  # 0: idle, 1: charging, 2: discharging
    caes_block_remaining = 0
    
    # Monthly definitions
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_marks = np.cumsum([0] + days_per_month) * 24
    month_per_hour = np.zeros(steps, dtype=int)
    for m in range(12):
        start, end = month_marks[m], month_marks[m+1]
        month_per_hour[start:end] = m

    # Optimized thermal scheduling
    for h in range(steps):
        hour = h % 24
        month = month_per_hour[h]
        
        if month in [4, 5, 6, 7]:  # Summer
            if 9 <= hour < 15:
                tp_seq[h] = 0.33
            elif 7 <= hour < 9 or 15 <= hour < 17:
                tp_seq[h] = 0.8
            else:
                tp_seq[h] = 1.0
        elif month in [10, 11, 0, 1]:  # Winter
            if 9 <= hour < 15:
                tp_seq[h] = 0.6
            elif 7 <= hour < 9 or 15 <= hour < 17:
                tp_seq[h] = 0.95
            else:
                tp_seq[h] = 1.0
        else:  # Transition months
            if 9 <= hour < 15:
                tp_seq[h] = 0.33
            elif 7 <= hour < 9 or 15 <= hour < 17:
                tp_seq[h] = 0.85
            else:
                tp_seq[h] = 1.0

    # Combined battery and CAES control
    soc = 0.5  # Battery initial SOC
    base_rate = 0.0005  # Base rate (50 kW)
    
    for h in range(steps):
        hour = h % 24
        month = month_per_hour[h]
        current_tp = tp_seq[h]
        
        # Seasonal SOC targets
        charge_target = 0.8 if month in [4, 5, 6, 7] else 0.7
        discharge_threshold = 0.2 if month in [10, 11, 0, 1] else 0.3
        
        # CAES state machine
        if caes_block_remaining > 0:
            caes_block_remaining -= 1
            if caes_state == 1:  # Charging
                caes_seq[h] = -0.33
                gas += 0.066
                hot += 0.033
                cold -= 0.0495
            elif caes_state == 2:  # Discharging
                caes_seq[h] = 0.86
                gas -= 0.172
                hot -= 0.086
                cold += 0.129
            
            # Clamp tank states
            gas = max(0.6, min(1.0, gas))
            hot = max(0.05, min(0.95, hot))
            cold = max(0.05, min(0.95, cold))
            
            # End block if completed
            if caes_block_remaining == 0:
                caes_state = 0
        else:
            # Try to start charging block (1-5 AM or 9 AM-3 PM)
            if (hour in [1,2,3,4] or (9 <= hour < 15)) and (h+3 < steps):
                gas_proj = gas + 4*0.066
                hot_proj = hot + 4*0.033
                cold_proj = cold - 4*0.0495
                
                if (0.6 <= gas_proj <= 1.0 and 
                    0.05 <= hot_proj <= 0.95 and 
                    0.05 <= cold_proj <= 0.95):
                    caes_state = 1
                    caes_block_remaining = 3
                    caes_seq[h] = -0.33
                    gas += 0.066
                    hot += 0.033
                    cold -= 0.0495
                    continue
            
            # Try to start discharging block (5-9 PM)
            if hour in [17,18,19,20] and (h+3 < steps):
                gas_proj = gas - 4*0.172
                hot_proj = hot - 4*0.086
                cold_proj = cold + 4*0.129
                
                if (0.6 <= gas_proj <= 1.0 and 
                    0.05 <= hot_proj <= 0.95 and 
                    0.05 <= cold_proj <= 0.95):
                    caes_state = 2
                    caes_block_remaining = 3
                    caes_seq[h] = 0.86
                    gas -= 0.172
                    hot -= 0.086
                    cold += 0.129
                    continue
        
        # Battery control
        charge_opportunity = (current_tp < 0.5) and (9 <= hour < 15) and (soc < charge_target)
        discharge_opportunity = (current_tp > 0.9) and ((7 <= hour < 9) or (17 <= hour < 21)) and (soc > discharge_threshold)
        
        if charge_opportunity:
            rate_scale = min(1.0, (charge_target - soc) * 2)
            battery_seq[h] = -base_rate * rate_scale
            soc += (-battery_seq[h]) * charge_factor
        elif discharge_opportunity:
            rate_scale = min(1.0, (soc - discharge_threshold) * 2)
            battery_seq[h] = base_rate * rate_scale
            soc -= battery_seq[h] * discharge_factor
        
        # Proactive SOC protection
        if soc < 0.15:
            battery_seq[h] = -min(base_rate, (0.9 - soc) / charge_factor)
            soc += (-battery_seq[h]) * charge_factor
        elif soc > 0.85:
            battery_seq[h] = min(base_rate, (soc - 0.1) / discharge_factor)
            soc -= battery_seq[h] * discharge_factor
        
        # Final SOC clamping
        soc = max(0.1, min(0.9, soc))

    return caes_seq.tolist(), battery_seq.tolist(), tp_seq.tolist()
# EVOLVE-BLOCK-END


def main():
    """
    主入口函数。调用演化块生成的策略，返回给评估器。
    返回的格式必须为字典，且字典的 key 不能改变。
    """
    caes_seq, battery_seq, tp_seq = generate_energy_strategy(steps=8760)

    return {"caes_seq": caes_seq, "battery_seq": battery_seq, "tp_seq": tp_seq}
