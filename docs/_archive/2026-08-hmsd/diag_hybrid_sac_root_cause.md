<!-- ARCHIVED: superseded by FS-HSAC; see docs/paper_outline_and_figures.md -->

# 彻底诊断：为何 seasonal_v1 的「SAC」会 eval_failed

日期：2026-08-20  
证据：`docs/diag_hybrid_sac_eval_fail.json`、`docs/diag_hybrid_sac_roll.json`、checkpoint 权重键、held-out 逐步回放。

---

## 0. 一句话结论

磁盘上的 `runs/seasonal_v1/*/sac_s0` **不是**论文里写的「混合 SAC (mode, magnitude)」，而是**旧版连续投影 SAC**（`caes_mean`/`caes_log_std` → `tanh` → 三段投影）。  
策略常输出 `u_caes = -1`（满功率放电）；若干小时后 Oracle 把放电安全区间收窄到例如 `[-0.69,-0.33]`，`-1` 非法；确定性评测下 64 次重采样是**同一个动作** → `NoSafeActionFound` → `eval_failed`。

这不是「混合动作设计本身被证伪」，而是：**旧权重 + 动态幅值区间 + 硬 GiveSafe 无 fallback + 确定性塌缩** 叠在一起。

---

## 1. 身份错位（最重要）

| 项 | 论文 / 当前默认代码 | 磁盘 `sac_s0` / `td3_s0` |
|----|---------------------|-------------------------|
| CAES 头 | `caes_mode_head` + `caes_mag_*` | `caes_mean` + `caes_log_std`（SAC）或 `caes_head`（TD3） |
| `parameterized_caes` | 默认 `True` | checkpoint 字段为 `None`（旧存盘） |
| 解码 | mode+mag → 合法带内幅值 | `tanh(z)` → `project_u_caes`（空隙→0） |
| 训练步 | — | `total_it ≈ 839745`（约 84 万步，训练跑完了） |

用当前默认 `HybridSAC()` **strict load 会直接失败**（缺 mode/mag 键）。评测脚本若在改默认之后跑，必须 `parameterized_caes=False` 或 `strict=False` 才能加载——说明矩阵与「新混合 SAC」叙事**不同代**。

TD3 同代：也是投影头；冬/过渡/夏 held-out 分别活了 13 / 5 / 76 h，与「投影连续在硬合法集上早死」一致。

---

## 2. 失败的物理机制（逐步回放）

### Winter held-out（`eval_start=3024000`）

- t=0..1：策略 `u_caes=-1`，GiveSafe **接受**（当时满放仍在安全区间内）。
- t≈7..9：一度落到 `u_caes=0`（idle）。
- **t=10 挂死**：确定性动作塌回 `u_caes=-1`。  
  Oracle：`u_caes=-1.0 不在放电安全幅值区间 [-0.6859, -0.33]`。  
  `mode_mask` 仍是 discharge/idle/charge 全开，但**幅值子区间**已不允许 -1。  
  64 次提议全是同一 `-1` → `NoSafeActionFound`。

### Transition held-out（`eval_start=10886400`）

- t=0..4：几乎步步 `u_caes=-1`，气体 SoC 从 ~0.81 降到 ~0.70。
- **t=5 挂死**：同样 `-1` 不在 `[-0.8953, -0.33]`。  
  此时 `mode_mask = {discharge: true, idle: false, charge: false}`——**只允许放电**，但必须是带内较弱放电；策略不会改幅值。

训练期 `safety_dataset.json`（冬）：484 条拒绝**全部**是 `caes_pressure_low`——长期在学「想放电但罐压/安全幅值不让」，与评测死因同族。

---

## 3. 为何「设计好了」仍会错（分层解释）

```text
设计层：mode+mag + mask + GiveSafe     ← 代码默认已朝这里改
         ≠
训练层：seasonal_v1 实际跑的是投影 SAC  ← 磁盘事实
         ≠
评测层：确定性 + 无 fallback + 动态幅值区间
```

1. **设计 ≠ 已训练**：提纲写的混合 SAC 尚未成为这批 `sac_s0` 的权重。  
2. **合法带 ≠ 当前安全幅值**：三段合法集 `[-1,-0.33]∪{0}∪[0.86,1]` 只是静态外壳；Oracle 每步还有更窄的放电/充电区间。投影到 `-1` 在静态合法，却可在动态区间非法。  
3. **GiveSafe 重采样在确定性评测下失效**：`deterministic=True` 时 `z=μ`、模式 argmax；方差塌缩时连 `deterministic=False` 也几乎同一动作（探测：`n_propose_unique=1`）。64 次 = 同一非法动作 ×64。  
4. **协议故意不救命**：`use_fallback=false` 时，找不到合法动作就记失败——这是论文协议，不是神秘 bug。

---

## 4. 和 HMSD / 双层 TD3 的关系

| 选项 | 是否被本诊断支持 |
|------|------------------|
| 「SAC 废了必须回 HMSD」 | **否**。挂的是**投影连续 SAC**，不是 mode+mag 混合 SAC。 |
| 「分层才能交卷」 | **未证明**。同协议下投影 TD3 也早停；HMSD 亦非三季满周。 |
| 「先重跑真正的 parameterized SAC」 | **是**。这是唯一对齐论文身份的下一步。 |

---

## 5. 根因清单（按优先级）

1. **P0 矩阵错代**：`seasonal_v1` 的 sac/td3 是投影消融代，不是混合动作代 → 不能支撑「混合 SAC 失败/成功」的论文结论。  
2. **P0 策略不尊重动态幅值区间**：输出贴带端点 `-1` / `+1`，Oracle 收窄后必拒。  
3. **P1 确定性评测 × 方差塌缩**：GiveSafe 重采样形同虚设。  
4. **P1 load API**：`HybridSAC.load` 不读 `parameterized_caes`，且 Actor 即使 `False` 仍注册 mode/mag 参数 → strict load 脆。  
5. **P2** 训练拒绝几乎全是 `caes_pressure_low`：奖励/探索未学会在收窄区间内降功率或切 idle。

---

## 6. 建议动作（工程，不改论文身份）

1. **重跑**三季 `parameterized_caes=True` 的 hybrid SAC（及混合 TD3），run 目录与旧投影 run **隔离命名**（例如 `sac_param_s0`），checkpoint 写入 `parameterized_caes: true`。  
2. **解码时钳制幅度**：`mag` 映射到 Oracle 给出的当前放电/充电安全区间，而不是整段 `[-1,-0.33]`。  
3. **评测 GiveSafe**：非法时允许对 **幅值/模式在 mask∩区间内** 做有限随机重采样（仍无规则 fallback），或对确定性策略做「投影到当前安全区间」再检查——需在论文里写清，避免被审成作弊。  
4. **修复 load**：按 checkpoint 字段重建 Actor；`parameterized_caes=False` 时不要注册多余 mode 头（或 `strict=False` + 断言键集合）。  
5. 旧 `sac_s0` 仅作 **C2 投影消融证据**（早死 + 端点放电），不要再写进「混合 SAC 主结果」。

---

## 7. 对你「感觉很奇怪」的直接回答

不是模块没接上，而是：

- 你以为在评「已设计好的混合 SAC」；  
- 实际在评「旧投影 SAC 学到的满功率放电习惯」；  
- 罐压一紧，Oracle 收窄幅值，确定性策略还死磕 `-1`；  
- 协议规定此时必须失败。

所以奇怪感来自 **叙事代际 ≠ 磁盘权重**，不是物理模型突然坏了，也不是必须立刻退回 HMSD。
