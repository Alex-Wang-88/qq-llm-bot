@echo off
chcp 65001 >nul
title qq-llm-bot 启动器
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_local.ps1"
set "taskCode=%errorlevel%"
if not "%taskCode%"=="0" (
    echo.
    echo [启动失败] 退出码 %taskCode%
    echo 原因已写入 data\logs\startup.log，最后几行：
    echo -----------------------------------------------
    powershell -NoProfile -Command "if (Test-Path '%~dp0data\logs\startup.log') { Get-Content '%~dp0data\logs\startup.log' -Tail 8 }"
    echo -----------------------------------------------
    echo.
    pause
    exit /b %taskCode%
)
echo.
echo [启动成功] 面板 http://127.0.0.1:6185
echo 日志目录 data\logs  ^(启动过程见 startup.log^)
timeout /t 8 >nul
exit /b 0
