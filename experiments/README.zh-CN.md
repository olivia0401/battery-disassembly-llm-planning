# 毕业论文实验套件

对应论文第 4 章的实验："**Verifiable Language-to-Skill Planning for Battery Disassembly in ROS2 via Episodic Retrieval and Two-Tier Safety Validation**"

## 测试集

`unified_test_suite.json` 里有 34 条命令，分布在 9 个类别（functional_primitive、functional_grasp_release、functional_skill_specific、functional_multistep、stress_ambiguous、stress_underspecified、stress_colloquial、stress_out_of_domain、stress_complex_reasoning）。标准答案在 `eval/reference_plans.json`（唯一权威来源，由 `eval/gen_reference_plans.py` 生成）：其中 18 条是 `should_pass`（机器人真实能力范围内可以完成的），16 条是 `should_block`（要么指代不清、要么超出能力范围、要么涉及机器人根本没有真实关节角度/场景定义的位置或物体——拒绝执行才是正确答案，不是功能缺失）。

## 实验设计

| 实验 | 问题 | 脚本 |
|---|---|---|
| **RQ1** | 组件消融——RAG/校验相比纯脚本硬编码基线，真的有用吗？（SB / LO / LV / LR / FS 五种配置） | `run_fast.py --rq 1` |
| **RQ2** | 两层校验器（Schema / Rule / Full）能不能拦住不安全或无效的计划？ | `run_fast.py --rq 2` |
| **RQ3** | RAG 记忆库大小的敏感性——检索多少个历史案例才够？ | `run_fast.py --rq 3` |
| **RQ4** | *（探索性）* 感知噪声鲁棒性——纯几何数学模拟，不是真实摄像头管线 | `run_rq4_perception_noise.py` |
| **RQ5** | 在 RQ1-3 里打分好/差的计划，放到真实 ROS2/MoveIt 上到底能不能安全执行？包含一项"纵深防御"探测：把 should_block 命令里未经校验、被模型瞎编出来的计划直接发给真实的 skill_server，看调度层能不能拦住文本层校验漏掉的东西。 | `run_rq5_real_execution.py` |

RQ1-3 共用同一批底层生成的计划（`run_fast.py` 里的 `run_planning_matrix`），所以必须通过 `run_fast.py` 一起跑，不要用那几个独立的旧脚本 `run_rq1_ablation.py` / `run_rq2_safety.py` / `run_rq3_memory.py`（这些是 `run_fast.py` 为了提速而取代的早期单实验实现；之所以保留，是因为 `run_fast.py` 仍然从 `run_rq2_safety.py` 里导入校验器的类）。

## 怎么跑

### 1. RQ1-3（会调用 LLM——用云端后端会产生费用）

```bash
cd experiments
python3 run_fast.py --rq all --leakfree --trials 5 --backend openai --concurrency 4
# 想免费/本地跑就用 --backend ollama（会慢很多，纯 CPU 大概每次调用 55-100 秒）
```

`--leakfree` 用的是 `eval/memory_split.json`（30 个 RAG 案例，已核实跟 34 条测试命令没有任何逐字重叠），而不是跟测试集完全一样的记忆库。断点续跑安全：中途中断重新执行会自动跳过已经完成的 `(command, task, config, trial)` 组合；如果某次调用失败、或者悄悄降级成非 LLM 的保底方案，**不会**被标记为完成，而是自动重试，不会污染数据集。

### 2. 分析

```bash
python -m eval.analyze       # -> 生成 eval/analysis_summary.json + 控制台摘要
python -m eval.build_workbook  # -> 生成 eval/Result_robot.xlsx
```

会输出 Wilson 95% 置信区间、McNemar 配对显著性检验（经 Holm-Bonferroni 校正）、7 类失败模式细分、留一命令交叉验证的稳定性，以及合并后的 RQ1×RQ2 因子表（RAG 开/关 × 校验强度）——RAG=off 时的 Schema/Rule 那两格会诚实地标注"未测试"，而不是编个数字出来，因为这个组合从来没有真的跑过。

### 3. RQ4（免费、离线，不需要 ROS2）

```bash
python3 run_rq4_perception_noise.py --trials 300
```

### 4. RQ5（需要真实运行的 ROS2 环境）

```bash
source /opt/ros/humble/setup.bash && source ../install/setup.bash
python3 run_rq5_real_execution.py                    # should_pass 命令：LO vs FS vs 标准答案
python3 run_rq5_real_execution.py --sources defense  # 纵深防御探测
```

每跑一条命令前都会重启整个 ROS2 环境（因为没有场景重置服务，只有重新启动才能保证每次都是干净的初始场景）。`use_rviz:=false` 已经自动处理好了——如果 RViz 是单独启动的，它会在这些重启过程中存活下来，不会被反复杀掉重开导致根本来不及看。

## 已知局限

- RQ5 每条命令、每个计划来源都只执行了一次（不是重复多次试验），所以它的成功率置信区间比较宽（17/18 对应 [74%, 99%]）——真实的运动规划（RRTConnect）本身存在一定的随机波动，这个设计没有覆盖到这一点。
- 每一行结果都记录了精确的模型 ID（例如 `openai:gpt-4o-mini`）和 ISO 时间戳——报告结果时请引用这个时间戳，因为服务商那边的模型别名可能在不改版本号的情况下悄悄更新底层模型。
- RQ1-3 里可能有一个 `(command, trial)` 组合缺失，如果某次 LLM 调用失败且还没重试过（`python -m eval.analyze` 会报出具体缺哪一条）；用同样的 `run_fast.py` 命令重新跑一次就会自动补上，不会重跑其他已完成的数据。
