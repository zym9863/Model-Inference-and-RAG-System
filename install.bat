@echo off
echo ========================================
echo RAG系统安装脚本 (Windows)
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

echo 检测到Python版本:
python --version
echo.

REM 创建虚拟环境
echo 正在创建虚拟环境...
python -m venv venv
if errorlevel 1 (
    echo 错误: 创建虚拟环境失败
    pause
    exit /b 1
)

REM 激活虚拟环境
echo 正在激活虚拟环境...
call venv\Scripts\activate.bat

REM 升级pip
echo 正在升级pip...
python -m pip install --upgrade pip

REM 安装依赖
echo 正在安装依赖包...
pip install -r requirements.txt
if errorlevel 1 (
    echo 错误: 依赖安装失败
    pause
    exit /b 1
)

REM 创建必要的目录
echo 正在创建目录结构...
if not exist "data" mkdir data
if not exist "logs" mkdir logs
if not exist "data\chromadb" mkdir data\chromadb
if not exist "data\sample_docs" mkdir data\sample_docs

echo.
echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 使用方法:
echo 1. 激活虚拟环境: venv\Scripts\activate.bat
echo 2. 运行系统: python main.py
echo 3. 运行示例: python examples\basic_usage.py
echo 4. 运行测试: python -m pytest tests\
echo.
echo 配置文件位置: config\config.yaml
echo 日志文件位置: logs\rag_system.log
echo.
pause
