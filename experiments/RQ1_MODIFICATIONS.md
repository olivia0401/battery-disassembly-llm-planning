# RQ1 实验修改说明

## 修改日期
2026-01-21

## 主要修改

### 1. 添加了第4个验证级别：RV (Rule-based Validation)

**新的验证级别**：
- **NV** (No Validation): 无验证，直接转发到执行
- **SV** (Schema-based Validation): 仅检查结构正确性（技能名称、参数类型）
- **RV** (Rule-based Validation): 仅检查序列约束（安全规则）**【新增】**
- **FV** (Full Validation): 完整验证（Schema + Rule-based）

### 2. 实现了 RuleBasedValidation 类

```python
class RuleBasedValidation:
    """Rule-based validation only - check sequence constraints but NOT schema."""

    def __init__(self, validator: Validator):
        """Initialize with Validator instance."""
        self.validator = validator

    def validate(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Rule-based validation: check sequence constraints only.
        Does NOT check schema (skill names, parameters).
        """
        # Only check sequence constraints, not schema
        violations = self.validator._check_sequence_constraints(plan)

        return {
            'valid': len(violations) == 0,
            'violations': violations
        }
```

### 3. 更新了实验配置

**之前**：
- 验证级别：3 个 (NV, SV, FR)
- 总测试数：35 × 3 × 3 = 315 次

**现在**：
- 验证级别：4 个 (NV, SV, RV, FV)
- 总测试数：35 × 4 × 3 = 420 次

### 4. 更新了默认参数

- 默认命令文件：`test_commands_minimal.json` → `unified_test_suite.json`
- 默认 trials：1 → 3

## 验证级别对比

| 级别 | Schema 检查 | Rule 检查 | 说明 |
|------|------------|-----------|------|
| NV   | ❌         | ❌        | 无任何验证 |
| SV   | ✅         | ❌        | 只检查技能名称、参数类型 |
| RV   | ❌         | ✅        | 只检查序列约束（如禁止连续release） |
| FV   | ✅         | ✅        | 完整验证（两层都检查） |

## 运行时间估算

- **完整运行** (--trials 3): 约 **28-30 分钟**
- 快速测试 (--trials 1): 约 **9-10 分钟**
- 中等测试 (--trials 2): 约 **18-20 分钟**

## 运行命令

```bash
cd /home/olivia/llms-ros2/experiments

# 完整运行（默认）
python3 run_rq1_safety.py

# 快速测试
python3 run_rq1_safety.py --trials 1

# 使用自定义命令文件
python3 run_rq1_safety.py --commands my_test_suite.json --trials 3
```

## 代码变更位置

1. **文件头部注释** (行 2-25): 更新了实验描述
2. **RuleBasedValidation 类** (行 139-157): 新增
3. **FullValidationWrapper 注释** (行 168): 更新说明
4. **validators 字典** (行 205-214): 添加 RV 级别
5. **参数解析** (行 375-378): 更新默认值

## 测试验证

语法检查通过：
```bash
✓ Syntax check passed
```

## 注意事项

1. RV 级别使用 `Validator._check_sequence_constraints()` 方法
2. 这是一个内部方法（以 `_` 开头），确保不会在未来版本中被移除
3. FV 级别现在使用统一的 validator 实例，与 RV 共享
4. 所有4个验证级别使用相同的接口，便于一致性测试
