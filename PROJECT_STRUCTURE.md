# LLM-ROS2 电池拆卸系统 - 项目结构

## 项目概述

基于LLM的ROS2机器人电池拆卸控制系统，支持通过自然语言指令控制Kinova Gen3机械臂执行拆卸任务。

## 目录结构

```
llms-ros2/
├── src/
│   ├── battery_dismantle_task/      # ROS2核心包
│   │   ├── battery_dismantle_task/  # Python模块
│   │   │   ├── skill_server.py      # 主技能服务器 (446行，已模块化)
│   │   │   ├── motion_executor.py   # 运动规划执行器 (197行)
│   │   │   ├── scene_manager.py     # Planning Scene管理 (102行)
│   │   │   ├── skill_handlers.py    # 高级技能处理 (139行)
│   │   │   ├── session.py           # Session数据类 (11行)
│   │   │   ├── visual_state_manager.py # 可视化管理
│   │   │   └── __init__.py
│   │   ├── config/                  # 配置文件
│   │   │   ├── waypoints.json       # 关节位置定义
│   │   │   ├── moveit/              # MoveIt配置
│   │   │   └── rviz/                # RViz配置
│   │   ├── launch/                  # 启动文件
│   │   │   └── fake_execution_complete.launch.py
│   │   └── package.xml
│   │
│   └── llm_agent/                   # LLM智能体
│       ├── executor.py              # 命令执行器 (258行)
│       ├── planner.py               # LLM规划器 (352行)
│       ├── llm_client.py            # LLM客户端
│       ├── rag_engine.py            # RAG检索引擎 (408行)
│       ├── validator.py             # 命令验证器
│       ├── web_ui.py                # Web界面 (360行，已合并)
│       ├── config/                  # 配置文件
│       │   ├── skills.json
│       │   └── safety.yaml
│       └── requirements.txt
│
├── FINAL_START.sh                   # 系统启动脚本
└── README.md                        # 项目说明
```

## 核心模块说明

### 1. ROS2机器人控制 (battery_dismantle_task)

#### skill_server.py - 主服务器
- **功能**: 接收LLM命令，调度技能执行
- **重构**: 从787行拆分为5个模块（446+197+139+102+11行）
- **模块化设计**:
  - `motion_executor.py`: 运动规划和轨迹执行
  - `scene_manager.py`: Planning Scene对象attach/detach
  - `skill_handlers.py`: 高级技能（grasp, release, dismantle）
  - `session.py`: 会话上下文数据

#### 支持的技能
- **原子技能**: moveTo, openGripper, closeGripper, approach, place, retreat
- **高级技能**: grasp, release, dismantle, sequence
- **管理技能**: selectObject, selectPlace

### 2. LLM智能体 (llm_agent)

#### planner.py - 任务规划
- **功能**: 将自然语言转换为机器人技能序列
- **特性**: 支持RAG检索历史经验

#### executor.py - 命令执行
- **功能**: 执行规划的技能序列
- **模式**: ROS2模式 / Mock模式

#### web_ui.py - Web界面
- **功能**: Gradio界面，用户交互
- **重构**: 合并了3个版本（web_ui.py, web_ui_v2.py, web_ui_fixed.py）
- **端口**: http://localhost:7862

## 快速启动

```bash
# 1. 启动完整系统
bash FINAL_START.sh

# 2. 在浏览器打开 Web UI
http://localhost:7862

# 3. 测试步骤
- 点击 'Initialize System'
- 输入: "Go to home position"
- 点击 'Execute'
- 观察 RViz 窗口
```

## 系统架构

```
用户输入 (Web UI)
    ↓
LLM Planner (自然语言 → 技能序列)
    ↓
Executor (发送ROS2命令)
    ↓
Skill Server (技能调度)
    ↓
Motion Executor (运动规划)
    ↓
MoveIt2 + ROS2 Control
    ↓
机器人执行
```

## 数据流

1. **命令格式** (JSON)
```json
{
  "schema": "llm_cmd/v1",
  "command_id": "cmd_001",
  "skill": "moveTo",
  "target": "HOME"
}
```

2. **反馈格式** (JSON)
```json
{
  "schema": "llm_fb/v1",
  "status": "success",
  "message": "Skill 'moveTo' completed",
  "command_id": "cmd_001"
}
```

## 重构改进

### 已完成
- ✅ 拆分skill_server.py (787行 → 5个文件<200行)
- ✅ 合并3个web_ui版本为1个
- ✅ 清理测试文件和备份
- ✅ 修复launch文件包路径
- ✅ 系统成功启动并运行

### 代码质量
- 所有核心工作文件 < 450行
- 模块化设计，职责清晰
- 适合工程部署

## 依赖项

### 系统依赖
- ROS2 Humble
- MoveIt2
- ros-humble-kortex-* (已安装在 /opt/ros/humble)

### Python依赖
- gradio
- openai (LLM API)
- chromadb (RAG)
- sentence-transformers (RAG)

## 配置文件

- `waypoints.json`: 机器人位姿定义
- `skills.json`: LLM技能描述
- `safety.yaml`: 安全限制
- `moveit/`: MoveIt2配置
- `rviz/`: RViz可视化配置

## 日志位置

- ROS2 Launch: `/tmp/ros2_launch.log`
- Web UI: `/tmp/webui_v2.log`
- 系统启动: `/tmp/final_start_output.log`

## 技术亮点

1. **模块化架构**: 核心代码拆分为独立模块，易于维护
2. **LLM集成**: 自然语言控制机器人
3. **RAG增强**: 基于历史经验优化任务规划
4. **MoveIt2**: 先进的运动规划
5. **Web界面**: 友好的用户交互

## 开发者指南

### 添加新技能
1. 在 `skill_handlers.py` 添加技能实现
2. 在 `skill_server.py` 的 `dispatch_skill()` 添加调度逻辑
3. 在 `config/skills.json` 添加技能描述

### 修改waypoints
编辑 `config/waypoints.json`，定义新的位姿或对象

### 调试
```bash
# 查看ROS2节点
ros2 node list

# 查看话题
ros2 topic list

# 查看skill_server日志
tail -f /tmp/ros2_launch.log | grep skill_server
```

## 许可证

Apache-2.0
