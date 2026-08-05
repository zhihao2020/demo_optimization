# P0 加固：无 GiveSafe 消融 + 多 seed

## 1. 无 GiveSafe（seed=0）

| 季节 | 方法 | 净现金流 J | 周 reward | 能量 SOC | FMU 失败 | 无效转移 |
|------|------|-----------:|----------:|:--------:|---------:|---------:|
| winter | hybrid_safe | 1.851e+07 | 128.1 | 是 | 0 | 0 |
| winter | hybrid_nosafe | 1.851e+07 | 128.1 | 是 | 0 | 0 |
| winter | ghtd3_safe | 1.831e+07 | 126.8 | 是 | 0 | 0 |
| winter | ghtd3_nosafe | 1.831e+07 | 126.8 | 是 | 0 | 0 |
| winter | b0 | 8.333e+06 | 67.5 | 是 | 0 | 0 |
| summer | hybrid_safe | 1.168e+07 | 83.7 | 是 | 0 | 0 |
| summer | hybrid_nosafe | 1.168e+07 | 83.7 | 是 | 0 | 0 |
| summer | ghtd3_safe | 1.118e+07 | 80.3 | 是 | 0 | 0 |
| summer | ghtd3_nosafe | 1.118e+07 | 80.3 | 是 | 0 | 0 |
| summer | b0 | -8.415e+04 | 13.3 | 是 | 0 | 0 |
| transition | hybrid_safe | 1.630e+07 | 113.6 | 是 | 0 | 0 |
| transition | hybrid_nosafe | 1.630e+07 | 113.6 | 是 | 0 | 0 |
| transition | ghtd3_safe | 1.618e+07 | 113.0 | 是 | 0 | 0 |
| transition | ghtd3_nosafe | 1.618e+07 | 113.0 | 是 | 0 | 0 |
| transition | b0 | 6.883e+06 | 58.6 | 是 | 0 | 0 |

## 2. 多 seed 净现金流 mean±std

| key | mean J | std | n |
|-----|-------:|----:|--:|
| summer|b0 | -8.415e+04 | 0.000e+00 | 3 |
| summer|ghtd3_safe | 1.118e+07 | 0.000e+00 | 3 |
| summer|hybrid_safe | 1.168e+07 | 0.000e+00 | 3 |
| transition|b0 | 6.883e+06 | 1.141e-09 | 3 |
| transition|ghtd3_safe | 1.618e+07 | 0.000e+00 | 3 |
| transition|hybrid_safe | 1.630e+07 | 2.281e-09 | 3 |
| winter|b0 | 8.333e+06 | 0.000e+00 | 3 |
| winter|ghtd3_safe | 1.831e+07 | 0.000e+00 | 3 |
| winter|hybrid_safe | 1.851e+07 | 0.000e+00 | 3 |

### 解读提示

- 若 `*_nosafe` 的 invalid/failure 上升或 J 下降，即可支撑 **GiveSafe 执行前过滤** 的贡献。
- 确定性策略下 seed 方差可能很小；若几乎为 0，正文写 *deterministic evaluation, seed for env init only*。


## 3. 动作噪声压力测试（冬周 Hybrid）

对策略动作加高斯噪声 (std=0.4) 并以 25% 概率随机 CAES 模式：

| 设置 | 净现金流 J | 周 reward | 无效转移 | 说明 |
|------|-----------:|----------:|---------:|------|
| noisy_nosafe | ~0 | 0.0 | >=1 | 轨迹崩溃 |
| noisy_givesafe | 1.24e+07 | 89.9 | 0 | GiveSafe 兜底 |

**结论**：名义评估关 GiveSafe 无差异；**扰动下 GiveSafe 是关键安全层**。

