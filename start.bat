@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] 没找到 .venv\Scripts\python.exe
    echo 请先在项目目录执行:  D:\python311\python.exe -m venv .venv
    pause
    exit /b 1
)

echo 正在启动 OCR 服务...
echo 打开浏览器访问: http://127.0.0.1:5000
echo 按 Ctrl+C 停止服务
echo.

".venv\Scripts\python.exe" app.py

pause
