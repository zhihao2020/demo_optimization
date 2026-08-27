@echo off
cd /d D:\xuzh\demo_optimization
set PYTHONUNBUFFERED=1
set PYTHONPATH=D:\xuzh\demo_optimization\src
set OPTIMAL_DEMO_CACHE=D:\xuzh\demo_optimization_cache
set OPTIMAL_DEMO_JOB_ID=tou2026_summer_fs_hsac_s0
set OPTIMAL_DEMO_TMP=D:\xuzh\demo_optimization_cache\tmp\tou2026_summer_fs_hsac_s0
set OPTIMAL_DEMO_FMU_ISOLATE=1
set CUDA_VISIBLE_DEVICES=0
set OPTIMAL_DEMO_DEVICE=cuda
set OPTIMAL_DEMO_TORCH_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
if not exist "%OPTIMAL_DEMO_TMP%" mkdir "%OPTIMAL_DEMO_TMP%"
if not exist "D:\xuzh\demo_optimization\runs\seasonal_tou2026\summer\fs_hsac_s0" mkdir "D:\xuzh\demo_optimization\runs\seasonal_tou2026\summer\fs_hsac_s0"
echo START %DATE% %TIME% > "D:\xuzh\demo_optimization\logs\tou2026_summer_fs_hsac_s0.log"
"D:\xuzh\demo_optimization\.venv\Scripts\python.exe" "D:\xuzh\demo_optimization\scripts\train_seasonal.py" --method fs_hsac --season summer --episodes 5000 --seed 0 --run-dir "D:\xuzh\demo_optimization\runs\seasonal_tou2026\summer\fs_hsac_s0" >> "D:\xuzh\demo_optimization\logs\tou2026_summer_fs_hsac_s0.log" 2>> "D:\xuzh\demo_optimization\logs\tou2026_summer_fs_hsac_s0.log.err"
echo EXIT %ERRORLEVEL% %DATE% %TIME% >> "D:\xuzh\demo_optimization\logs\tou2026_summer_fs_hsac_s0.log"
