#!/bin/bash
# 监控实验进度

echo "=== 实验运行状态 ==="
echo ""

# 检查进程
echo "Running Processes:"
ps aux | grep "run_rq" | grep -v grep | awk '{print "  PID " $2 ": " $11 " " $12 " " $13}'
echo ""

# 检查最新的log文件
echo "=== 最新日志 ==="
for rq in rq1 rq2 rq3; do
    latest_log=$(ls -t results/${rq}_run_*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ]; then
        echo ""
        echo "--- $rq ($(basename $latest_log)) ---"
        tail -5 "$latest_log"
    fi
done

echo ""
echo "=== 最新结果文件 ==="
ls -lth results/rq*_results_*.csv 2>/dev/null | head -3 | awk '{print $9 " (" $5 ", " $6 " " $7 " " $8 ")"}'
