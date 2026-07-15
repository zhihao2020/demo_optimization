# C_ref 跨季节重标定

- 旧 C_ref（单周 3 种子）: 96199.1266
- 新 C_ref（季节周 P95）: 156539.8354
- 相对变化 >5%: True
- 配置已更新: True
- 样本步数: 1008

场景周起点见 `scripts/calibrate_reward_seasonal.py` 中 `SEASONAL_STARTS`。
若差异不显著，保持冻结旧值；否则以季节 P95 回写 `reward_config.yaml`。
