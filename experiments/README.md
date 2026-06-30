# Experimental Suite for Dissertation

This directory contains automated experiments to generate results for Section 4 of the dissertation: **"Verifiable Language-to-Skill Planning for Battery Disassembly in ROS2 via Episodic Retrieval and Two-Tier Safety Validation"**

## Quick Start

```bash
# 1. Ensure OpenAI API key is set
export OPENAI_API_KEY="your-key-here"

# 2. Run all experiments (2-4 hours)
bash run_all_experiments.sh

# 3. View results
cat analysis/summary_report.txt
```

## Experimental Design

### RQ1: Ablation Study

**Research Question**: Does RAG-based experience retrieval improve task planning accuracy compared to baseline scripted control and pure LLM planning?

**Configurations**:
- **Scripted Baseline**: Keyword matching to predefined skill sequences
- **LLM-Only**: GPT-4 planning without RAG retrieval
- **Full RAG**: GPT-4 + ChromaDB experience retrieval + validation

**Test Protocol**:
- 50 commands (15 preset, 15 paraphrased, 20 variant)
- 3 configurations × 5 trials = 750 tests
- Metrics: M1 (success rate), M2 (latency), generalization ability

**Run**:
```bash
python3 run_rq1_ablation.py
```

**Expected Results**:
- Scripted: High accuracy on preset, poor on variants
- LLM-Only: Moderate accuracy, risk of hallucination
- Full RAG: Best overall, especially on variants

---

### RQ2: Memory Size Impact

**Research Question**: What is the optimal RAG memory size for battery disassembly tasks?

**Configurations**:
- k=0: No memory (same as LLM-Only)
- k=10: Small memory
- k=20: Medium memory
- k=35+: Full available memory

**Test Protocol**:
- Same 50 commands from RQ1
- 4 memory sizes × 5 trials = 1000 tests
- Metrics: Planning accuracy vs memory size, retrieval time

**Run**:
```bash
python3 run_rq2_memory.py
```

**Expected Results**:
- Performance saturates around k=20-30
- Diminishing returns beyond saturation point
- Retrieval time increases linearly with k

---

### RQ3: Safety Validation Impact

**Research Question**: Does the two-tier validation system improve reliability?

**Configurations**:
- **No Validation**: Plans executed directly without checks
- **Schema-Only**: Skill name and parameter type validation
- **Full Validation**: Schema + safety constraints

**Test Protocol**:
- 50 safe commands + 20 unsafe commands
- 3 validation levels × 5 trials = 1050 tests
- Metrics: M3 (safety violations), M4 (false positives), M5 (overhead)

**Run**:
```bash
python3 run_rq3_safety.py
```

**Expected Results**:
- No Validation: Accepts unsafe commands (high risk)
- Schema-Only: Catches malformed skills, misses semantic errors
- Full Validation: Catches both, minimal false positives

---

## Test Commands

50 test commands in `test_commands.json`:

**Preset (15)**: Direct skill-level commands
```
"Go to home position"
"Grasp the top cover bolts"
"Open the gripper"
```

**Paraphrased (15)**: Rephrased versions of known tasks
```
"Navigate to the home configuration"
"Pick up the bolts from the top cover"
"Activate gripper opening"
```

**Variant (20)**: Novel combinations requiring generalization
```
"Remove the battery cover bolts"
"Execute complete bolt removal sequence"
"Perform full battery dismantling procedure"
```

## Results Structure

```
experiments/
├── results/                      # Raw experimental data
│   ├── rq1_results_TIMESTAMP.csv
│   ├── rq1_results_TIMESTAMP.json
│   ├── rq2_results_TIMESTAMP.csv
│   ├── rq2_results_TIMESTAMP.json
│   ├── rq3_results_TIMESTAMP.csv
│   └── rq3_results_TIMESTAMP.json
│
└── analysis/                     # Processed results
    ├── table_rq1.tex            # LaTeX table for dissertation
    ├── table_rq2.tex
    ├── table_rq3.tex
    └── summary_report.txt       # Human-readable summary
```

## Data Analysis

After experiments complete:

```bash
# Generate LaTeX tables and summary
python3 analysis/analyze_results.py

# View summary
cat analysis/summary_report.txt

# Copy LaTeX tables to dissertation
cp analysis/table_rq*.tex /path/to/dissertation/tables/
```

## Metrics Definitions

**M1 - Success Rate**: Percentage of commands successfully planned and validated

**M2 - Planning Latency**: Time from command input to validated plan output
- Planning time: LLM API call
- Retrieval time: RAG database lookup (RQ2)
- Validation time: Safety checks (RQ3)

**M3 - Safety Violation Rate**: Percentage of unsafe commands NOT caught by validator

**M4 - False Positive Rate**: Percentage of safe commands incorrectly rejected

**M5 - Validation Overhead**: Additional time cost of validation layer

## Dissertation Integration

### Section 4.1: RQ1 Results

```latex
\subsection{RQ1: System Configuration Comparison}

Table \ref{tab:rq1_results} presents the ablation study comparing three system configurations across 750 test trials...

\input{tables/table_rq1.tex}

Our results demonstrate that the Full RAG system achieves XX\% success rate, outperforming both the Scripted Baseline (YY\%) and LLM-Only configuration (ZZ\%)...
```

### Section 4.2: RQ2 Results

```latex
\subsection{RQ2: RAG Memory Size Analysis}

Figure \ref{fig:memory_saturation} illustrates the relationship between RAG memory size and planning accuracy...

\input{tables/table_rq2.tex}

Performance saturates at approximately k=XX cases, suggesting...
```

### Section 4.3: RQ3 Results

```latex
\subsection{RQ3: Safety Validation Effectiveness}

Table \ref{tab:rq3_results} compares validation strategies...

\input{tables/table_rq3.tex}

The two-tier validation system successfully caught XX\% of unsafe commands while maintaining only Y\% false positive rate...
```

## Configuration

### API Keys

```bash
# OpenAI (required for LLM planning)
export OPENAI_API_KEY="sk-..."

# Alternative: Use local Ollama
# Install: https://ollama.ai
# Modify planner.py to use Ollama endpoint
```

### Experiment Parameters

Edit script parameters in each `run_rqN_*.py`:

```python
# Number of trials per test case
num_trials=5  # Increase to 10 for more robust statistics

# Random seed for reproducibility
random_seed=42

# Memory sizes (RQ2)
memory_sizes=[0, 10, 20, 35]

# Validation levels (RQ3)
validation_levels=['no_validation', 'schema_only', 'full_validation']
```

## Troubleshooting

### OpenAI API Rate Limits

If you hit rate limits, add delays:

```python
# In run_rqN_*.py, after each LLM call:
import time
time.sleep(1)  # 1 second delay between requests
```

### Missing Dependencies

```bash
cd ../src/llm_agent
pip install -r requirements.txt
pip install pandas  # For analysis
```

### Partial Results

If experiments are interrupted, results are auto-saved every 50 trials. Resume by:

1. Comment out completed experiments in `run_all_experiments.sh`
2. Re-run remaining experiments
3. Analysis script automatically uses latest results

## Validation Checklist

Before submitting to Frontiers:

- [ ] All 3 experiments completed (750 + 1000 + 1050 = 2800 trials)
- [ ] No API errors or exceptions in logs
- [ ] Success rates are reasonable (not 0% or 100%)
- [ ] LaTeX tables compile in dissertation
- [ ] Results match Research Questions in Section 3
- [ ] Statistical significance tested (t-tests, ANOVA)
- [ ] Figures generated from CSV data
- [ ] Discussion section updated with findings

## Expected Timeline

- **RQ1**: 45-90 minutes (750 LLM calls)
- **RQ2**: 60-120 minutes (1000 LLM calls)
- **RQ3**: 60-120 minutes (1050 LLM calls, including unsafe tests)
- **Analysis**: 5 minutes
- **Total**: 2-4 hours (depends on API speed)

## Citation

If you use this experimental framework in your research:

```bibtex
@software{xu2024rag_experiments,
  author = {Xu, Olivia},
  title = {Experimental Suite for RAG-Enhanced LLM Robot Control},
  year = {2024},
  url = {https://github.com/olivia0401/RAG-validation-LLMs-ROS2/tree/main/experiments}
}
```

## Support

If experiments fail:
1. Check OpenAI API key is valid
2. Ensure all dependencies installed
3. Review error logs in console output
4. Open issue: https://github.com/olivia0401/RAG-validation-LLMs-ROS2/issues

---

**Good luck with your publication!** 🎓
