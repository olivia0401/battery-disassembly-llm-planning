"""
Executor - ROS2 or Mock
Provides fine-grained control and state query capabilities
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
import json
import time
from typing import Dict, List, Optional


class Executor:
    """Task Executor - Communicates with robot via ROS2 topics"""

    def __init__(self, use_ros: bool = True):
        """
        Initialize executor

        Args:
            use_ros: Whether to use real ROS2 (False for mock)
        """
        self.use_ros = use_ros
        self.node = None
        self.command_pub = None
        self.feedback_sub = None
        self.joint_state_sub = None
        self.last_feedback = None
        self.last_joint_state = None
        self.current_gripper_state = "unknown"

        if use_ros:
            self._init_ros()

    def _init_ros(self):
        """Initialize ROS2 node"""
        if not rclpy.ok():
            rclpy.init()

        self.node = rclpy.create_node('llm_executor')
        self.command_pub = self.node.create_publisher(String, '/llm_commands', 10)
        self.feedback_sub = self.node.create_subscription(
            String,
            '/llm_feedback',
            self._feedback_callback,
            10
        )
        self.joint_state_sub = self.node.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10
        )

        print("✅ ROS2 executor initialized")

        # Wait for topic connections to establish
        print("⏳ Waiting for skill_server connection...")
        time.sleep(3.0)

        # Spin once to process any pending callbacks
        rclpy.spin_once(self.node, timeout_sec=0.1)
        print("✅ Ready to send commands")

    def _feedback_callback(self, msg):
        """Receive feedback"""
        self.last_feedback = msg.data
        print(f"   📥 Feedback: {msg.data}")

    def _joint_state_callback(self, msg: JointState):
        """Receive joint state updates"""
        self.last_joint_state = msg
        # Track gripper state from joint positions
        if 'robotiq_85_left_knuckle_joint' in msg.name:
            idx = msg.name.index('robotiq_85_left_knuckle_joint')
            gripper_pos = msg.position[idx]
            self.current_gripper_state = "open" if gripper_pos < 0.4 else "closed"

    def execute(self, plan: Dict, timeout: float = 120.0) -> Dict:
        """
        执行计划

        Args:
            plan: 计划字典
            timeout: 每个动作的超时时间(秒)

        Returns:
            执行结果 {"success": bool, "executed": int, "failed": int, "log": [...]}
        """
        actions = plan['plan']
        results = {
            "success": True,
            "executed": 0,
            "failed": 0,
            "log": []
        }

        print(f"\n🚀 Executing plan ({len(actions)} steps)...")

        for i, action in enumerate(actions, 1):
            skill = action['name']
            target = action['params'].get('target', None)

            if target:
                print(f"\n📍 Step {i}/{len(actions)}: {skill}(target={target})")
            else:
                print(f"\n📍 Step {i}/{len(actions)}: {skill}()")

            if self.use_ros:
                success = self._execute_ros(action, timeout)
            else:
                success = self._execute_mock(action)

            # 记录结果
            log_entry = {
                "step": i,
                "action": action,
                "success": success,
                "timestamp": time.time()
            }
            results['log'].append(log_entry)

            if success:
                results['executed'] += 1
                print(f"   ✅ Success")
            else:
                results['failed'] += 1
                results['success'] = False
                print(f"   ❌ Failed")
                # 是否继续?
                break

            time.sleep(0.5)  # 短暂延迟

        print(f"\n📊 Execution Summary:")
        print(f"   Total: {len(actions)}")
        print(f"   Executed: {results['executed']}")
        print(f"   Failed: {results['failed']}")

        return results

    def _execute_ros(self, action: Dict, timeout: float) -> bool:
        """通过ROS2执行"""
        # 构造命令 - 必须包含schema字段
        command = {
            "schema": "llm_cmd/v1",
            "skill": action['name']
        }

        # 添加target参数（如果存在）
        if 'target' in action['params']:
            command['target'] = action['params']['target']

        # 发布命令
        msg = String()
        msg.data = json.dumps(command)
        self.command_pub.publish(msg)
        print(f"   📤 Published command: {msg.data}")

        # 等待反馈
        self.last_feedback = None
        start_time = time.time()

        while self.last_feedback is None:
            rclpy.spin_once(self.node, timeout_sec=0.1)

            if time.time() - start_time > timeout:
                print(f"   ⏱️  Timeout after {timeout}s")
                return False

        # 解析JSON反馈: {"status": "success", "message": "...", "timestamp": ...}
        try:
            feedback = json.loads(self.last_feedback)
            return feedback.get("status") == "success"
        except (json.JSONDecodeError, KeyError):
            # 兼容简单字符串反馈
            return self.last_feedback == "success"

    def _execute_mock(self, action: Dict) -> bool:
        """Mock execution with realistic failure injection"""
        time.sleep(0.2)  # Simulate execution time

        # Deterministic failure based on action hash
        skill_name = action['name']
        target = action['params'].get('target', '')
        action_str = f"{skill_name}_{target}"
        action_hash = hash(action_str) % 100

        # Skill-specific failure rates (降低以获得合理成功率)
        failure_thresholds = {
            'moveTo': 1,           # 1% failure (降低)
            'grasp': 1,            # 1% failure (降低)
            'release': 0,          # 0% failure (基础操作不应失败)
            'unscrew': 2,          # 2% failure (降低)
            'disconnect': 1,       # 1% failure (降低)
            'inspect': 0,          # 0% failure
            'openGripper': 0,      # 0% failure (基础操作不应失败)
            'closeGripper': 0,     # 0% failure (基础操作不应失败)
            'waitForStabilization': 0,  # Never fails
            'rotateGripper': 1     # 1% failure
        }

        threshold = failure_thresholds.get(skill_name, 3)

        if action_hash < threshold:
            failure_reasons = {
                'grasp': 'gripper_slip',
                'moveTo': 'trajectory_planning_failed',
                'unscrew': 'bolt_stuck',
                'disconnect': 'connector_tight'
            }
            reason = failure_reasons.get(skill_name, 'execution_timeout')
            print(f"   ❌ Mock execution FAILED: {reason}")
            return False

        print(f"   ✅ Mock execution")
        return True

    def get_current_state(self) -> Optional[Dict]:
        """
        Query current robot state

        Returns:
            Dict with joint positions, gripper state, or None if not available
        """
        if not self.use_ros or self.last_joint_state is None:
            return None

        # Spin once to get latest state
        rclpy.spin_once(self.node, timeout_sec=0.1)

        if self.last_joint_state is None:
            return None

        # Extract arm joint positions (first 7 joints)
        arm_joints = {}
        gripper_joints = {}

        for i, name in enumerate(self.last_joint_state.name):
            if i < len(self.last_joint_state.position):
                pos = self.last_joint_state.position[i]
                if 'joint_' in name:
                    arm_joints[name] = round(pos, 4)
                elif 'robotiq' in name:
                    gripper_joints[name] = round(pos, 4)

        return {
            "arm_joints": arm_joints,
            "gripper_joints": gripper_joints,
            "gripper_state": self.current_gripper_state,
            "timestamp": time.time()
        }

    def print_current_state(self):
        """Print current robot state to console"""
        state = self.get_current_state()

        if state is None:
            print("❌ No state available")
            return

        print("\n📊 Current Robot State:")
        print(f"   Gripper: {state['gripper_state']}")
        print(f"   Arm Joints:")
        for joint, pos in state['arm_joints'].items():
            print(f"      {joint}: {pos:.4f} rad")
        print(f"   Timestamp: {state['timestamp']:.2f}\n")

    def shutdown(self):
        """Shutdown executor"""
        if self.use_ros and self.node:
            self.node.destroy_node()
            # Note: Don't call rclpy.shutdown() as other ROS2 nodes may still be using it
            # rclpy.shutdown()


# 测试
def test():
    # Mock测试
    executor = Executor(use_ros=False)

    plan = {
        "plan": [
            {"step": 1, "name": "moveTo", "params": {"target": "HOME"}},
            {"step": 2, "name": "grasp", "params": {"target": "TopCoverBolts"}}
        ]
    }

    results = executor.execute(plan)
    print(f"\nResults: {results['success']}")


if __name__ == "__main__":
    test()
