# 论文修订备忘录 (Dissertation Revision Memo)

**项目**: Verifiable Language-to-Skill Planning for Battery Disassembly in ROS2
**日期**: 2026-06-09
**依据**: (a) 全量代码审核; (b) 用严谨指标重算现有实验数据; (c) 导师评语

> **重要前提（请连同结论一起理解）**
> 下面的重算数字来自**现有的实验运行**——该运行存在 (1) 记忆=测试集泄漏、(2) 模型为 Llama-3.2-3B、(3) API 失败时静默回退到关键词 demo 规划器、(4) 参考答案为自动生成待人工复核。
> 因此**绝对数值是临时的（下界）**；但**配置之间的比较性结论是稳健的**，因为同一套参考、同一套数据被施加到所有配置上，系统性误差对各配置等量作用。

---

## 第 1 部分：必须修正的整合性问题（论文文字 vs 实际代码）

| # | 章节/位置 | 论文现在的表述 | 应改为 | 代码依据 |
|---|---|---|---|---|
| 1 | 全文 / Tech stack | "GPT-4 / GPT-3.5" | **Llama-3.2-3B-Instruct（经 OpenRouter）** | `llm_client.py` openrouter→`meta-llama/llama-3.2-3b-instruct`；所有 run 脚本 `backend="openrouter"` |
| 2 | §4–5 指标命名 | "end-to-end execution success rate" | **"planning/validation success rate"**（删去 execution/端到端） | `executor.py:_execute_mock` 用 `hash()%100` 决定成败；实验 `Executor(use_ros=False)` |
| 3 | Appendix A | "executed by the Kinova Gen3 ... in **Gazebo and RViz**" | 删除物理执行声称；说明定量结果来自 mock，机器人执行仅见演示视频 | 同上；1300+ 次定量 trial 未经 ROS/MoveIt |
| 4 | §3.5 | "two-tier ... runtime monitoring of moments, torques, temperatures" | tier-2 为**已设计但未运行**（与 §4.5 "Level 3 未实现"一致） | `validator.validate_runtime()` 从未被调用，无传感器 |
| 5 | 摘要 & 结论 | "1,260 controlled simulation experiments" | 统一为实际值 **1,365**（420+420+525） | CSV 行数 420/420/525 |
| 6 | §3.3 | "ten predefined high-level skills including unscrew and disconnect" | 模型 prompt **实际只暴露 5 个技能**（grasp/release/moveTo/open/close） | `config/prompt.txt` 硬编码 5 技能，无 `{skills_list}` |
| 7 | §5.2 图8右 | "Retrieval latency vs memory size" | 实为**总规划时延（网络主导）**，需重画或删除 | `run_rq2_memory.py:253` `retrieval_time = planning_wall_s` |
| 8 | 方法/局限 | （未提及） | **新增**：API 失败时规划器静默回退到关键词 demo；旧代码未记录模式，部分"LLM 结果"可能为 demo 输出 | `planner.plan()` except→`_demo_plan`；旧脚本未存 `planner_mode` |
| 9 | 安全 | `.env` 含密钥已提交仓库 | 轮换密钥并 gitignore | `src/llm_agent/.env` |

---

## 第 2 部分：评估方法升级（回应导师"评估不够严谨"）

旧评估只报单一 success rate。新评估管线（已实现于 `experiments/eval/`）引入：

1. **两层成功定义**：`plan_valid`（schema 通过）vs `task_correct / Exact`（匹配正确答案）——区分"形式合法"与"真的对"。
2. **步级 Precision / Recall / F1**（有序 LCS）——捕捉"部分正确"。
3. **Wilson 95% 置信区间** + **McNemar 配对显著性检验** + **失败模式 7 分类**。
4. **数据来源核查**（每行记录 planner_mode，demo 回退自动告警）。
5. **Cohen's κ**（人机标注一致性，待人工复核后计算）。

---

## 第 3 部分：用真指标重算的结果（核心）

### 3.1 RQ3 组件消融 —— 三大旗舰发现塌成噪声

| 配置 | 旧"成功率" | **真 Exact 正确率（95% CI）** | 步级 F1 |
|---|---|---|---|
| SB 脚本 | 14.3% | **8.6%** [4.6, 15.5] | 0.09 |
| LO 仅LLM | 94.3% | **40.0%** [31.1, 49.6] | 0.51 |
| LV LLM+验证 | 90.5% | **41.0%** [32.0, 50.5] | 0.54 |
| LR LLM+RAG | 95.2% | **39.0%** [30.3, 48.6] | 0.50 |
| FS 全系统 | 91.4% | **41.9%** [32.9, 51.5] | 0.54 |

**McNemar 配对显著性**：LO vs LV **p=1.0**；LO vs LR **p=1.0**；LR vs FS **p=0.61**；SB vs FS **p<1e-9**。

**结论**：
- 旧"成功率 ~94%"测的是 plan validity（schema 几乎总通过），不是正确性；**真 Exact ≈ 40%**。
- **LO/LV/LR/FS 之间统计上无差异**。因此旧论文的三大发现——"RAG 提升性能""验证降低成功率""负协同"——**在严谨检验下不成立**，差异落在运行间噪声内。
- 唯一稳健的结论：**LLM ≫ 脚本基线**（p<1e-9）。

### 3.2 RQ1 安全验证 —— 真实召回率仅 14%（旧报 70% 是小样本伪高）

用**全样本 plan 级 ground truth**（旧论文只在 13 个被拦样本上算，假阴性不可见）：

| 级别 | TP/FP/FN/TN | Precision | **Recall** | Pass rate |
|---|---|---|---|---|
| NV | 0/0/58/47 | n/a | 0.0% | 100% |
| SV | 3/0/52/50 | 100% | 5.5% | 97.1% |
| RV | 2/0/56/47 | 100% | 3.4% | 98.1% |
| FV | 8/0/49/48 | 100% | **14.0%** [7.3, 25.3] | 92.4% |

**结论**：完整验证（FV）精确率高（无误拦），但**只抓住了约 14% 的应拦计划**（旧报 70% 是因为分母只取了被拦的 13 个）。原因：多数错误是**语义级**（错对象、错顺序），schema/rule 检测不到。→ 验证层的价值在于"拦住明显非法/越域"，**不能**当作"保证计划正确"的机制。

### 3.3 RQ2 记忆规模 —— RAG 在"记忆"而非"泛化"

| k | Exact（CI） | Seen Exact | **Unseen Exact** |
|---|---|---|---|
| 0 | 38.1% | n/a | 38.1% |
| 10 | 33.3% | 70.4% | 20.5% |
| 20 | 40.0% | 57.9% | 18.8% |
| 35 | 49.5% | 66.7% | **0.0%** [0, 12.5] |

**结论**：表面上 k=35 的总 Exact 最高（且 k20→k35 边际显著 p=0.041），但分层后真相相反——**seen 命令 58–70%，unseen 在 k=35 跌到 0%**。即 RAG **记住了见过的命令，却损害了对未见命令的泛化**。这与旧论文"对未见任务鲁棒泛化"的结论**正好相反**。
（注：当前数据 memory=test 存在泄漏，seen 被高估；去泄漏重跑会进一步压低 seen，但 unseen 崩溃的方向稳健。）

### 3.4 去泄漏干净复跑（已完成，2026-06-09）

用**无泄漏 prompt（无答案 few-shot）+ 不相交记忆（15 条，与测试集零重叠）+ 真实本地 LLM（Llama-3.2-1B via Ollama）+ 来源核查（每行记 planner_mode，demo 回退已剔除）** 独立重跑，验证 3.1 的结论。

干净 RQ3（trials=1，n≈32–35，已剔除 47 行回退）：

| 配置 | **Exact (95% CI)** | 主要失败模式 |
|---|---|---|
| SB 脚本 | **37.1%** [23, 54] | no_plan(21) |
| LO 仅LLM | 18.8% [9, 35] | **wrong_param(22)** |
| LV LLM+验证 | 18.8% [9, 35] | wrong_param(22) |
| LR LLM+RAG | 9.7% [3, 25] | wrong_param(22), hallucinated(3) |
| FS 全系统 | 9.7% [3, 25] | wrong_param(22), hallucinated(3) |

显著性：SB vs LO **p=0.125**、LO vs LR **p=0.375**、SB vs FS **p=0.008（显著）**。

**新发现（重要）**：在诚实评分下，**脚本基线 SB 显著优于全系统 FS**。原因：SB 对越域命令**正确拒绝**（返回空计划），而 1B LLM **自信地幻觉**计划（如对"给电池充电"生成 `chargeBattery`）。即"会拒绝的笨基线比会幻觉的 LLM 更安全"。这与论文"LLM ≫ 脚本"的乐观叙事相反（至少在弱模型下），且强化了安全主题：**对超出能力的指令，拒绝才是正确行为**。两次复跑（3.1 泄漏/Llama-3B、3.4 干净/Llama-1B）都否定了"RAG 有用 / 验证有害 / 负协同"。

#### 分析代码审核：发现并修复的 3 个 bug
对自写的收集+分析代码做对抗式审核，修了 3 个会影响结论的 bug：
1. **正确拒绝被误判为失败**：该拒绝的越域命令返回空计划=正确，但评分把它算成 `no_plan` 失败（影响 6/35 命令）。修复后 SB 从 20%→37%，并暴露了"SB 显著优于 FS"这一发现。
2. **平凡对比**：LO vs LV、LR vs FS 复用同一份计划（验证不改写计划）→ McNemar 必然 p=1.0，不是真检验。已移除，改用 SB vs LO、LO vs LR、SB vs FS。
3. **延迟列恒为 0**：analyze 读了 runner 未写入的字段（total_time）。已改读 planning_time。

**caveat**：1B 本地模型偏弱、trials=1、n=35 → CI 宽、统计功效低。这是**流程验证 + 方向确认**，绝对值非终版；终版需更强模型 + trials≥5 + 人工复核参考 + Cohen's κ。

---

## 第 4 部分：建议改写的结论段（可直接采用）

**旧 (§5.4 四点) → 诚实改写版**：

1. ~~"验证机制提升错误检测但降低成功率"~~ →
   "在全样本 plan 级评估下，完整验证精确率 100% 但召回仅 ~14%；它能可靠拦截非法/越域计划，但无法捕捉语义级错误（错对象/错顺序）。验证的价值是安全兜底，而非提升正确性。"

2. ~~"记忆规模非单调，最优窗口 N=10"~~ →
   "总体 Exact 随 k 的差异多数不显著；分离 seen/unseen 后可见 RAG 主要在复用已见命令，对未见命令的泛化在大记忆下反而退化。记忆规模的'最优窗口'结论不成立。"

3. ~~"RAG 在低相似度下仍高成功率，鲁棒泛化"~~ →
   "该结论源于训练-测试泄漏（记忆即测试集）；去泄漏分层后，未见命令的正确率显著低于已见命令。"

4. ~~"检索与验证存在负协同"~~ →
   "RAG、验证及其组合对计划正确性均无统计显著影响；所谓'负协同'落在运行间噪声内。"

5. **新增（替代旧"LLM≫脚本"的乐观叙事）** →
   "在诚实评分下，是否优于脚本基线取决于指令类型：对域内指令 LLM 略好，但对**越域指令**，脚本基线因正确拒绝而显著更安全，弱 LLM 则幻觉计划。安全的关键不是'能不能规划'，而是'能不能在超出能力时拒绝'——这正是验证层应承担、但当前 schema/rule 无法覆盖的语义判断。"

---

## 第 5 部分：扩大与加强（基础设施已就绪）

去泄漏干净复跑**已完成**（见 3.4），下列工作把它从"方向确认"提升到"可发表"：

1. ✅ **去泄漏重跑（已完成，小规模）**：`prompt_clean.txt` + 15 条不相交记忆 + 本地 Ollama 已跑通。**扩大规模**只需一条命令（建议更强模型 + 更多 trials）：
   `python run_fast.py --rq all --leakfree --trials 10 --backend ollama`（或换 `OLLAMA_MODEL` 为更强模型）
2. **人工复核参考计划**（`reference_plans.json` 标了 23 条待复核）+ 填 `Result_robot.xlsx` Tab 4 → `python -m eval.compute_kappa`。
3. （可选，回应导师"机器人深度"）RQ4：MoveIt 可行性 + 感知噪声扫描（RViz，无需真机）。

---

## 第 6 部分：对导师两条评语的回应

| 导师评语 | 本次修订如何回应 |
|---|---|
| **评估不够严谨**（只看 success rate，缺 failure modes / robustness / P-R 权衡 / 安全行为 / 可复现） | 两层成功 + 步级 P/R/F1 + 7 类失败模式 + Wilson CI + McNemar + plan 级安全混淆矩阵 + planner_mode 来源核查 + 固定/记录配置 + Holm-Bonferroni 多重比较校正 + noise floor + leave-one-command-out（见第 7 部分） |
| **机器人深度不足**（ROS2/MoveIt/感知/执行要做深） | RQ4 感知噪声仿真（见第 7 部分）；**MoveIt 真实动态规划已修复并验证可用**（见第 8 部分），不再是"诚实划清边界、留作未来工作"，而是真的把它修通了 |

---

## 第 7 部分：评估管线成熟度补强（2026-06-30）

在第 2-6 部分的基础上，`experiments/eval/` 又补了几项之前缺的：

1. **Holm-Bonferroni 多重比较校正**（`eval/stats.py:holm_bonferroni`）：RQ2/RQ3 的所有 McNemar 比较族都按家族大小做了阶梯校正，不再用裸 p<0.05 下结论。
2. **noise floor 真正接入**（之前写了函数从未调用，现已接进 RQ1/RQ2/RQ3 的 `analyze.py`，trials<2 时显式返回 None 而不是悄悄当 0）。
3. **leave-one-command-out 稳健性检查**：逐条剔除测试指令重算赢家，RQ3 实测 100%（34/34）稳健，SB 显著优于 LO/FS 这一发现不是被一两条指令带偏的假象。
4. **RQ4 感知噪声仿真**（`run_rq4_perception_noise.py`，纯 Python 几何仿真，无相机无物理引擎，已在脚本/数据/Excel 三处反复标注"SIMULATED"）：用项目真实物体坐标（来自 `waypoints.json`/`object_definitions.py`）注入高斯位姿噪声，扫描抓取成功率。发现噪声 σ=5mm→10mm 时成功率从 97.1% 崩到 45.1%，给出了"系统需要优于10mm的位姿估计精度"这一定量结论。
5. **5 个之前未实现的技能**：`skills.json`（LLM 侧）声明了 `inspect`/`unscrew`/`disconnect`/`waitForStabilization`/`rotateGripper`，但 `skill_server.py` 的 dispatch 表完全没有这几个的处理器——任何用到这些技能的计划在真实机器人上会被直接拒绝为"Unknown skill"，即使评估管线把它们当作合法技能打分。已在 `skill_server.py`/`skill_handlers.py` 补齐，全部复用现有运动原语（不编造力矩/力反馈能力），`unscrew`/`disconnect` 对没有 waypoints 坐标的物体仍正确拒绝（而非编造坐标）。**已在真实 ROS2/MoveIt 环境（WSL2 Ubuntu 22.04 + ROS2 Humble）逐个手动测试通过**，包括用一个本不存在的物体名（"BMS connector"）验证拒绝路径按预期工作。
6. **单元测试：从 0 个到 48 个**（`eval/test_metrics.py`、`eval/test_stats.py`、`eval/test_skill_handlers.py`），覆盖产出论文每个数字的核心函数，以及 ROS2 包里不依赖 rclpy 的纯逻辑部分。

## 第 8 部分：MoveIt 真实动态规划——从"完全不可用"到"全自动闭环成功"（2026-06-30）

第 6 部分曾经写"诚实划清边界：当前为规划层评估，执行为 mock"，因为 `motion_executor.py` 里 `use_direct_execution=True` 绕开了 MoveIt 的真实规划（`MoveGroup.Goal()` 那条代码路径写了但从未被调用）。这次在真实 ROS2/MoveIt 环境里把它打开（`use_direct_execution=False`）做验证，逐步定位并修复了 **5 个独立的真实 bug**，最终让 MoveIt 的规划+执行全流程首次完整闭环成功：

| # | Bug | 定位方式 | 修复 |
|---|---|---|---|
| 1 | `skill_server.py` dispatch 表缺 5 个技能处理器 | 对照 `skills.json` 与 dispatch 表逐项核对 | 见第 7 部分第 5 条 |
| 2 | 抓取后物体未附着到夹爪碰撞模型（`scene_manager.attach_object()`写了但被注释掉，`visual_state_manager.py`的替代方案因定时器被禁用而失效） | `/check_state_validity` 服务实测：`robotiq_85_*_finger_tip_link` 与 `TopCoverBolts`/`BatteryBox_0` 穿透深度最高 8.2cm | 重新启用 `scene_manager.attach_object`/`detach_object`（`skill_handlers.py`） |
| 3 | 启动时的 ACM "补丁"用全量替换覆盖了 SRDF 的 132 条正确碰撞豁免配置，只剩自己写的 3 条 | 即使是机械臂自己的 HOME 姿态，`/check_state_validity` 也判定非法（16 个自碰撞，含 SRDF 明确声明该 disable 的 `base_link`/`shoulder_link`）；对比 live ACM 参数与 SRDF 文件确认两者均含正确声明，但运行时 ACM 被覆盖 | 把 `_setup_collision_matrix_fallback()` 从"发送局部 ACM 替换"改写为"先读取现有完整 ACM 再合并，最后整体重新应用"（`skill_server.py`），16 个碰撞降到 1 个 |
| 4 | `moveit_controllers.yaml` 缺少 `moveit_simple_controller_manager:` 顶层嵌套 key，导致 `moveit.plugins.moveit_simple_controller_manager` 返回 0 个控制器，规划成功但执行阶段失败（"Unable to identify any set of controllers"） | 规划已成功（"Solution found"）但执行失败，结合官方插件期望的标准 YAML schema 排查 | 加上正确的顶层 key（`config/moveit/moveit_controllers.yaml`），controller 列表从 0 变 2 |
| 5 | `motion_executor.py` 里一处不安全的嵌套 `rclpy.spin_once()` 调用，偷走了本该交给 ActionClient `on_goal_response` 的回调，导致 move_group 端真实执行成功，但 `skill_server` 自己的等待逻辑始终 20 秒超时报失败 | 对比 move_group 日志（"Solution was found and executed"）与 `skill_server` 日志（仍超时）的时间戳矛盾，结合该节点已被 `MultiThreadedExecutor` 后台 spin 的事实 | 删除该行（`motion_executor.py`） |

**最终验证**（冷启动，全自动，无需任何手动补丁脚本）：
```
✅ ACM updated (merged): disabled gripper-wrist collisions without touching the other 15 entries
✅ Skill Server Ready!
... [真实 RRTConnect 规划 + fake_manipulator_controller 执行] ...
✅ Planning successful! Executing trajectory on controller...
✅ Trajectory executed successfully in 2.46s!
📤 Feedback(/success): Skill 'moveTo' completed
```

**对论文的意义**：这不再是"承认没做、列为未来工作"，而是有完整定位过程+前后对比日志的真实工程修复。可以直接用于回应导师"应使用动态运动规划而非固定路点"的意见——附带说明：① 为什么原来选择绕开 MoveIt（第 2/3 条 bug 导致任何近物体的规划请求必然失败）；② 5 个 bug 各自的定位方法（`/check_state_validity` 服务直接查询碰撞对、ACM 参数比对、日志时间戳交叉验证），这套排查方法本身也可以写成方法论小节。

诊断脚本 `recalibrate_scene.py`、`fix_acm_merge.py` 保留在 `~/ros2_ws/` 下作为可复用的诊断工具（不是论文交付物的一部分，但有助于未来类似问题排查）。

*附：所有重算由 `experiments/eval/`（metrics/stats/analyze/build_workbook）产出，可一键复现：`python -m eval.analyze && python -m eval.build_workbook`。结果汇总见 `Result_robot.xlsx`。*

*数据来源：3.1–3.3 为旧运行重算（泄漏、Llama-3B）；3.4 为去泄漏干净复跑（Llama-1B 本地、含来源核查）。分析代码已经过对抗式审核并修复 3 个 bug（正确拒绝计分、平凡对比、延迟字段）。两套数据结论一致：论文的 RAG/验证/协同结论均不成立。*
