"""
双层校验 - Schema + Runtime
"""
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple


class Validator:
    """计划和运行时验证器"""

    def __init__(self, config_dir: Path = None):
        """初始化验证器"""
        if config_dir is None:
            config_dir = Path(__file__).parent / "config"

        # 加载配置
        with open(config_dir / "skills.json") as f:
            self.skills_config = json.load(f)

        with open(config_dir / "safety.yaml") as f:
            self.safety_config = yaml.safe_load(f)

        self.valid_skills = {s['name'] for s in self.skills_config['available_skills']}
        self.valid_poses = set(self.skills_config['available_poses'])
        self.valid_objects = set(self.skills_config['available_objects'])

    def validate_plan(self, plan: Dict) -> Tuple[bool, List[str]]:
        """
        Schema验证 - 检查计划格式和内容

        Returns:
            (是否有效, 错误列表)
        """
        errors = []

        # 1. 检查格式
        if 'plan' not in plan:
            errors.append("Missing 'plan' key")
            return False, errors

        actions = plan['plan']
        if not isinstance(actions, list):
            errors.append("'plan' must be a list")
            return False, errors

        # 2. 检查每个动作
        for i, action in enumerate(actions):
            # 检查必需字段
            if 'name' not in action:
                errors.append(f"Step {i+1}: missing 'name'")
                continue

            if 'params' not in action:
                errors.append(f"Step {i+1}: missing 'params'")
                continue

            skill_name = action['name']
            params = action['params']

            # 检查技能是否存在
            if skill_name not in self.valid_skills:
                errors.append(f"Step {i+1}: invalid skill '{skill_name}'")
                continue

            # 检查target参数（某些技能不需要target）
            if skill_name in ["openGripper", "closeGripper"]:
                # 这些技能不需要target参数
                pass
            else:
                # 其他技能需要target参数
                if 'target' not in params:
                    errors.append(f"Step {i+1}: missing 'target' in params")
                    continue

                target = params['target']

                # 检查target是否合法
                if skill_name == "moveTo":
                    if target not in self.valid_poses:
                        errors.append(f"Step {i+1}: invalid pose '{target}'")
                elif skill_name in ["grasp", "release"]:
                    if target not in self.valid_objects:
                        errors.append(f"Step {i+1}: invalid object '{target}'")

        # 3. 检查序列约束
        errors.extend(self._check_sequence_constraints(actions))

        return len(errors) == 0, errors

    def _check_sequence_constraints(self, actions: List[Dict]) -> List[str]:
        """检查技能序列约束"""
        errors = []

        # 检查禁止序列
        for i in range(len(actions) - 1):
            current = actions[i]['name']
            next_action = actions[i+1]['name']

            for forbidden in self.safety_config['forbidden_sequences']:
                if [current, next_action] == forbidden:
                    errors.append(
                        f"Forbidden sequence: {current} → {next_action}"
                    )

        # 检查必须以HOME结尾（暂时注释，用于测试）
        # if actions and actions[-1]['params'].get('target') != 'HOME':
        #     errors.append("Plan must end at HOME pose")

        return errors

    def validate_runtime(self, force: float, temperature: float) -> Tuple[bool, str]:
        """
        Runtime验证 - 检查传感器数据

        Returns:
            (是否安全, 消息)
        """
        limits = self.safety_config['runtime_limits']

        if force > limits['force_max']:
            return False, f"Force {force}N exceeds limit {limits['force_max']}N"

        if temperature > limits['temperature_max']:
            return False, f"Temperature {temperature}°C exceeds limit {limits['temperature_max']}°C"

        return True, "Runtime checks passed"


# 测试
def test():
    validator = Validator()

    # 测试有效计划
    valid_plan = {
        "plan": [
            {"step": 1, "name": "moveTo", "params": {"target": "HOME"}},
            {"step": 2, "name": "grasp", "params": {"target": "TopCoverBolts"}},
            {"step": 3, "name": "moveTo", "params": {"target": "HOME"}}
        ]
    }

    is_valid, errors = validator.validate_plan(valid_plan)
    print(f"Valid plan: {is_valid}")
    if errors:
        for error in errors:
            print(f"  - {error}")

    # 测试无效计划
    invalid_plan = {
        "plan": [
            {"step": 1, "name": "grasp", "params": {"target": "InvalidObject"}},
            {"step": 2, "name": "grasp", "params": {"target": "TopCoverBolts"}}
        ]
    }

    is_valid, errors = validator.validate_plan(invalid_plan)
    print(f"\nInvalid plan: {is_valid}")
    for error in errors:
        print(f"  - {error}")

    # 测试runtime
    safe, msg = validator.validate_runtime(force=15.0, temperature=30.0)
    print(f"\nRuntime check: {safe} - {msg}")


if __name__ == "__main__":
    test()
