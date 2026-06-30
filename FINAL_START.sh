#!/bin/bash
# 最终启动脚本 - 完全清理并重新启动

echo "=========================================="
echo "步骤 1: 彻底清理所有进程"
echo "=========================================="

# 停止所有相关进程
killall -9 python3 2>/dev/null
killall -9 rviz2 2>/dev/null
killall -9 ros2 2>/dev/null

sleep 3

# 停止 ROS2 daemon
ros2 daemon stop 2>/dev/null
sleep 2

# 再次确认
pkill -9 -f "ros2 launch" 2>/dev/null
pkill -9 -f "web_ui" 2>/dev/null
pkill -9 -f "rviz" 2>/dev/null

sleep 2

echo "✅ 所有进程已清理"
echo ""

echo "=========================================="
echo "步骤 2: 启动 ROS2 系统"
echo "=========================================="

cd /home/olivia/llms-ros2
source /opt/ros/humble/setup.bash
source install/setup.bash

# 在后台启动 ROS2
nohup ros2 launch battery_dismantle_task fake_execution_complete.launch.py > /tmp/ros2_launch.log 2>&1 &
ROS2_PID=$!

echo "✅ ROS2 启动中 (PID: $ROS2_PID)"
echo "   等待 20 秒让系统完全初始化..."

sleep 20

# 检查 skill_server
if ros2 node list 2>/dev/null | grep -q skill_server; then
    echo "✅ skill_server 已就绪"
else
    echo "❌ skill_server 未启动！"
    exit 1
fi

echo ""
echo "=========================================="
echo "步骤 3: 初始化关节状态"
echo "=========================================="

# 发布初始关节状态 (HOME位置 - 与waypoints.json一致)
ros2 topic pub --once /joint_states sensor_msgs/msg/JointState \
"{header: {stamp: {sec: 0, nanosec: 0}, frame_id: 'base_link'}, \
name: ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7', \
'robotiq_85_left_knuckle_joint', 'robotiq_85_right_knuckle_joint'], \
position: [0.0, 0.2618, 3.14159, -2.2689, 0.0, 0.9599, 1.5708, 0.0, 0.0], \
velocity: [], effort: []}" > /dev/null 2>&1

sleep 2

# 验证关节状态
if ros2 topic echo /joint_states --once 2>&1 | grep -q "position:"; then
    echo "✅ 关节状态已初始化"
else
    echo "⚠️  关节状态可能未正确初始化"
fi

echo ""
echo "=========================================="
echo "步骤 4: 启动 Web UI"
echo "=========================================="

cd /home/olivia/llms-ros2/src/llm_agent
source .venv/bin/activate

# Load OPENROUTER_API_KEY (and any other secrets) from .env — copy .env.example to .env first
if [ -f .env ]; then
    set -a
    source .env
    set +a
else
    echo "❌ .env not found in src/llm_agent/. Copy .env.example to .env and add your OPENROUTER_API_KEY."
    exit 1
fi

# 在后台启动 Web UI
nohup python3 web_ui.py > /tmp/webui_v2.log 2>&1 &
WEBUI_PID=$!

echo "✅ Web UI 启动中 (PID: $WEBUI_PID)"
echo "   等待 5 秒..."

sleep 5

# 检查端口
if ss -tlnp 2>/dev/null | grep -q ":7862"; then
    echo "✅ Web UI 运行在 http://localhost:7862"
else
    echo "❌ Web UI 未能启动在端口 7862"
fi

echo ""
echo "=========================================="
echo "🎉 启动完成！"
echo "=========================================="
echo ""
echo "请在浏览器打开: http://localhost:7862"
echo ""
echo "测试步骤:"
echo "1. 点击 'Initialize System'"
echo "2. 输入: Go to home position"
echo "3. 点击 'Execute'"
echo "4. 观察 RViz 窗口和 Console Output"
echo ""
echo "日志文件:"
echo "  ROS2:   /tmp/ros2_launch.log"
echo "  Web UI: /tmp/webui_v2.log"
echo ""
echo "=========================================="
