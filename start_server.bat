@echo off
echo 正在启动 Tendency 数据分析服务...
echo 如果这是首次运行，系统将安装必要的依赖包

REM 检查是否安装了依赖包
pip show flask >nul 2>&1
if %errorlevel% neq 0 (
    echo 正在安装必要的依赖包...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo 安装依赖包失败。请手动运行: pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo 依赖包安装成功！
)

REM 检查是否安装了python-dotenv
pip show python-dotenv >nul 2>&1
if %errorlevel% neq 0 (
    echo 正在安装python-dotenv...
    pip install python-dotenv
    if %errorlevel% neq 0 (
        echo 安装python-dotenv失败。请手动运行: pip install python-dotenv
        pause
        exit /b 1
    )
    echo python-dotenv安装成功！
)

echo 正在启动服务...
echo 服务启动后，请在浏览器中访问: http://127.0.0.1:5000

python app.py

pause
