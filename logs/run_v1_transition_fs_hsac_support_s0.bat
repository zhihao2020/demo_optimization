@echo off
cd /d D:\xuzh\demo_optimization
set PYTHONUNBUFFERED=1
set PYTHONPATH=D:\xuzh\demo_optimization\src
set OPTIMAL_DEMO_CACHE=D:\xuzh\demo_optimization_cache
set OPTIMAL_DEMO_JOB_ID=seasonal_transition_fs_hsac_support_s0
set OPTIMAL_DEMO_TMP=D:\xuzh\demo_optimization_cache\tmp\seasonal_transition_fs_hsac_support_s0
set OPTIMAL_DEMO_FMU_ISOLATE=1
set FS_HSAC_NO_FEAS=1
if not exist "%OPTIMAL_DEMO_TMP%" mkdir "%OPTIMAL_DEMO_TMP%"
if not exist "D:\xuzh\demo_optimization\runs\seasonal_v1\transition\fs_hsac_support_s0" mkdir "D:\xuzh\demo_optimization\runs\seasonal_v1\transition\fs_hsac_support_s0"
echo START %DATE% %TIME% > "D:\xuzh\demo_optimization\logs\seasonal_v1_transition_fs_hsac_support_s0.log"
"D:\xuzh\demo_optimization\.venv\Scripts\python.exe" "D:\xuzh\demo_optimization\scripts\train_seasonal.py" --method fs_hsac --season transition --episodes 5000 --seed 0 --run-dir "D:\xuzh\demo_optimization\runs\seasonal_v1\transition\fs_hsac_support_s0" >> "D:\xuzh\demo_optimization\logs\seasonal_v1_transition_fs_hsac_support_s0.log" 2>> "D:\xuzh\demo_optimization\logs\seasonal_v1_transition_fs_hsac_support_s0.log.err"
echo EXIT %ERRORLEVEL% %DATE% %TIME% >> "D:\xuzh\demo_optimization\logs\seasonal_v1_transition_fs_hsac_support_s0.log"
