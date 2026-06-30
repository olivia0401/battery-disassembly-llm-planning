#!/bin/bash
# RQ1 实验监控脚本

LOG_FILE="results/rq1_run_20260121_220533.log"

echo "=========================================="
echo "RQ1 实验监控"
echo "=========================================="
echo ""

# 检查进程是否在运行
if ps aux | grep -q "[r]un_rq1_safety.py"; then
    echo "✓ 实验正在运行中..."
    echo ""
else
    echo "✗ 实验已停止"
    echo ""
fi

# 显示最新进度
echo "最新进度："
echo "----------------------------------------"
tail -20 "$LOG_FILE" | grep -E "(\[.*\]|Validation:|Trial|Results saved)"
echo ""

# 统计进度
TOTAL_TESTS=420
COMPLETED=$(grep -c "✓ PASS\|✗ REJECT\|✓ CAUGHT\|✗ MISSED" "$LOG_FILE" 2>/dev/null || echo 0)
PERCENTAGE=$(echo "scale=1; $COMPLETED * 100 / $TOTAL_TESTS" | bc 2>/dev/null || echo "0")

echo "=========================================="
echo "完成度: $COMPLETED / $TOTAL_TESTS ($PERCENTAGE%)"
echo "=========================================="
echo ""

# 预计剩余时间
if [ "$COMPLETED" -gt 0 ]; then
    START_TIME=$(stat -c %Y "$LOG_FILE")
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    AVG_TIME_PER_TEST=$(echo "scale=2; $ELAPSED / $COMPLETED" | bc)
    REMAINING_TESTS=$((TOTAL_TESTS - COMPLETED))
    REMAINING_SECONDS=$(echo "$AVG_TIME_PER_TEST * $REMAINING_TESTS" | bc | cut -d. -f1)
    REMAINING_MINUTES=$((REMAINING_SECONDS / 60))

    echo "已运行时间: $((ELAPSED / 60)) 分钟"
    echo "预计剩余时间: $REMAINING_MINUTES 分钟"
fi

echo ""
echo "使用方法："
echo "  watch -n 10 ./monitor_rq1.sh  # 每10秒自动刷新"
echo "  tail -f $LOG_FILE             # 实时查看日志"
