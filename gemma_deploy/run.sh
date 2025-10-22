#!/bin/bash

# 设置Python环境路径
PYTHON_PATH="/root/miniconda3/envs/llama/bin/python"

# 设置环境变量，跳过vllm版本检查
export DISABLE_VERSION_CHECK=1

# 服务配置
SERVICE=${1:-server}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 检查Python环境是否存在
if [ ! -e "$PYTHON_PATH" ]; then
    echo "❌ Python环境不存在: $PYTHON_PATH"
    echo "请检查conda环境是否正确激活"
    echo "尝试查找Python:"
    ls -la /root/miniconda3/envs/llama/bin/python* 2>/dev/null || echo "未找到Python文件"
    exit 1
fi

echo "🐍 使用Python环境: $PYTHON_PATH"

if [ "$SERVICE" = "stop" ]; then
    pkill -f "python start_server.py"
    pkill -f "python start_local.py"
    echo "服务已停止"
elif [ "$SERVICE" = "local" ]; then
    nohup $PYTHON_PATH start_local.py > "local_${TIMESTAMP}.log" 2>&1 &
    echo "本地服务已启动，日志: local_${TIMESTAMP}.log"
    echo "实时日志: tail -f local_${TIMESTAMP}.log"
else
    nohup $PYTHON_PATH start_server.py > "server_${TIMESTAMP}.log" 2>&1 &
    echo "服务器已启动，日志: server_${TIMESTAMP}.log"
    echo "实时日志: tail -f server_${TIMESTAMP}.log"
fi

# 使用示例:
# ./run.sh          # 启动服务器 (使用绝对路径Python)
# ./run.sh server   # 启动服务器  
# ./run.sh local    # 启动本地服务
# ./run.sh stop     # 停止所有服务
#
# 注意: 脚本会自动使用 /root/miniconda3/envs/llama/bin/python
