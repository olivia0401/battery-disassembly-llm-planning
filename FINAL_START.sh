#!/bin/bash
# Launch the full system: ROS 2 / MoveIt stack + skill server, then the Gradio web UI.
# Run from anywhere:  bash FINAL_START.sh
# Requires: ROS 2 Humble, a built workspace (colcon build -> install/), and
# src/llm_agent/.env (copy from .env.example and add your API key).

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$REPO_ROOT/src/llm_agent"
LOG_DIR="${LOG_DIR:-/tmp}"
ROS_LOG="$LOG_DIR/ros2_launch.log"
WEBUI_LOG="$LOG_DIR/webui.log"
WEBUI_PORT=7862

echo "=========================================="
echo "Step 1: stop any previous run of this project"
echo "=========================================="
# Only this project's processes -- never a blanket kill of python3/ros2.
pkill -f "ros2 launch battery_dismantle_task" 2>/dev/null
pkill -f "llm_agent/web_ui.py" 2>/dev/null
pkill -f "python3 web_ui.py" 2>/dev/null
sleep 2
echo "[OK] Previous processes stopped"
echo ""

echo "=========================================="
echo "Step 2: start the ROS 2 stack"
echo "=========================================="
if [ ! -f /opt/ros/humble/setup.bash ]; then
    echo "[FAIL] /opt/ros/humble/setup.bash not found - install ROS 2 Humble first."
    exit 1
fi
if [ ! -f "$REPO_ROOT/install/setup.bash" ]; then
    echo "[FAIL] $REPO_ROOT/install/setup.bash not found - run 'colcon build' in $REPO_ROOT first."
    exit 1
fi
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "$REPO_ROOT/install/setup.bash"

cd "$REPO_ROOT" || exit 1
nohup ros2 launch battery_dismantle_task fake_execution_complete.launch.py > "$ROS_LOG" 2>&1 &
ROS2_PID=$!
echo "[OK] ROS 2 starting (PID: $ROS2_PID); waiting 20s for initialisation..."
sleep 20

if ros2 node list 2>/dev/null | grep -q skill_server; then
    echo "[OK] skill_server is ready"
else
    echo "[FAIL] skill_server did not start - see $ROS_LOG"
    exit 1
fi
echo ""

echo "=========================================="
echo "Step 3: publish the initial joint state (HOME, matches waypoints.json)"
echo "=========================================="
ros2 topic pub --once /joint_states sensor_msgs/msg/JointState \
"{header: {stamp: {sec: 0, nanosec: 0}, frame_id: 'base_link'}, \
name: ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7', \
'robotiq_85_left_knuckle_joint', 'robotiq_85_right_knuckle_joint'], \
position: [0.0, 0.2618, 3.14159, -2.2689, 0.0, 0.9599, 1.5708, 0.0, 0.0], \
velocity: [], effort: []}" > /dev/null 2>&1
sleep 2

if ros2 topic echo /joint_states --once 2>&1 | grep -q "position:"; then
    echo "[OK] Joint states initialised"
else
    echo "[WARN] Joint states may not be initialised correctly"
fi
echo ""

echo "=========================================="
echo "Step 4: start the web UI"
echo "=========================================="
cd "$AGENT_DIR" || exit 1
# Optional virtualenv: use src/llm_agent/.venv if present.
if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
else
    echo "[FAIL] $AGENT_DIR/.env not found. Copy .env.example to .env and set LLM_BACKEND and its API key."
    exit 1
fi

nohup python3 web_ui.py > "$WEBUI_LOG" 2>&1 &
WEBUI_PID=$!
echo "[OK] Web UI starting (PID: $WEBUI_PID); waiting 5s..."
sleep 5

if ss -tln 2>/dev/null | grep -q ":$WEBUI_PORT"; then
    echo "[OK] Web UI running at http://localhost:$WEBUI_PORT"
else
    echo "[FAIL] Web UI is not listening on port $WEBUI_PORT - see $WEBUI_LOG"
fi

echo ""
echo "=========================================="
echo "Started. Open http://localhost:$WEBUI_PORT"
echo "=========================================="
echo "Try: click 'Initialize System', enter 'Go to home position', click 'Execute',"
echo "then watch RViz and the console output."
echo ""
echo "Logs:"
echo "  ROS 2:  $ROS_LOG"
echo "  Web UI: $WEBUI_LOG"
