@echo off
echo ========================================
echo RAG系统示例运行脚本
echo ========================================
echo.

REM 检查虚拟环境是否存在
if not exist "venv\Scripts\activate.bat" (
    echo 错误: 虚拟环境不存在，请先运行 install.bat
    pause
    exit /b 1
)

REM 激活虚拟环境
call venv\Scripts\activate.bat

echo 选择要运行的示例:
echo 1. 基本使用示例 (basic_usage.py)
echo 2. 文档处理示例 (document_processing.py)
echo 3. 交互式RAG系统 (main.py)
echo 4. 运行测试 (test_rag_system.py)
echo.

set /p choice="请输入选择 (1-4): "

if "%choice%"=="1" (
    echo 正在运行基本使用示例...
    python examples\basic_usage.py
) else if "%choice%"=="2" (
    echo 正在运行文档处理示例...
    python examples\document_processing.py
) else if "%choice%"=="3" (
    echo 正在启动交互式RAG系统...
    python main.py --mode interactive
) else if "%choice%"=="4" (
    echo 正在运行测试...
    python -m pytest tests\ -v
) else (
    echo 无效选择，请重新运行脚本
)

echo.
pause
