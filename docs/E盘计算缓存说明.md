# E: 盘计算缓存（减轻 D: 占用）

## 布局

| 路径 | 用途 |
|------|------|
| `E:\optimal_demo_cache\runs` | 训练/评估/基准输出（checkpoint、轨迹 CSV） |
| `E:\optimal_demo_cache\tmp` | TEMP/TMP（FMU/系统临时文件） |
| `E:\optimal_demo_cache\pycache` | `PYTHONPYCACHEPREFIX` |
| `E:\optimal_demo_cache\torch` | `TORCH_HOME` |
| `E:\optimal_demo_cache\pip` | `PIP_CACHE_DIR` |
| `E:\optimal_demo_cache\fmu_work` | 预留 FMU 工作副本 |
| `E:\optimal_demo_cache\logs` | 长任务日志 |

仓库内 `D:\Code\0622\optimal_demo\runs` 通过 **junction** 指向 `E:\optimal_demo_cache\runs`，因此相对路径 `runs/...` 无需改代码即可写到 E:。

## 一次性设置

```powershell
cd D:\Code\0622\optimal_demo
# 先结束占用 runs 的 python 进程
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
powershell -ExecutionPolicy Bypass -File scripts/setup_e_drive_cache.ps1
```

会：

1. 把现有 `runs` **复制**到 E:  
2. 将 D 上原目录改名为 `runs_d_backup_*`  
3. 创建 `runs` → `E:\optimal_demo_cache\runs` 联接  
4. 设置 **用户级** 环境变量（新终端生效）

确认 junction 正常后，可删除 D: 上的 `runs_d_backup_*` 释放空间：

```powershell
Remove-Item -Recurse -Force D:\Code\0622\optimal_demo\runs_d_backup_*
```

当前会话也可：

```powershell
. E:\optimal_demo_cache\session_env.ps1
```

## 代码入口

- `src/config/paths.py`：`apply_process_cache_env()` / `resolve_run_dir()`  
- `scripts/run_full_benchmark.py`、`scripts/train_ghtd3.py` 启动时自动应用  

覆盖路径：

```powershell
$env:OPTIMAL_DEMO_CACHE = "E:\optimal_demo_cache"
$env:OPTIMAL_DEMO_RUNS = "E:\optimal_demo_cache\runs"
$env:OPTIMAL_DEMO_TMP = "E:\optimal_demo_cache\tmp"
```

## 注意

- **源码与 FMU 仍在 D:**（`src/`、`data/*.fmu`），只把大体量产物放到 E:。  
- 备份盘 / 同步工具若跟随 junction，注意是否复制真实数据。  
- C: 上的 `%LOCALAPPDATA%\Temp` 在设置 User `TEMP` 后新进程会改走 E:；已打开的旧进程仍可能用旧 TEMP。
