#!/bin/bash
# capture_logs.sh – Extract signal quality logs from CryptoBot v4
# Usage: ./capture_logs.sh [futures|spot] [lines] [output_file]

set -e

MODE="${1:-futures}"
LINES="${2:-500}"
OUTPUT="${3:-signal_analysis_${MODE}_$(date +%Y%m%d_%H%M%S).log}"

LOG_DIR="/root/cryptobot_v3/logs"
LOG_FILE="${LOG_DIR}/${MODE}_bot.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "ERROR: Log file not found: $LOG_FILE"
    echo "Make sure the bot is running and mode is correct (spot/futures)."
    exit 1
fi

echo "Capturing last $LINES lines from $LOG_FILE"
echo "Saving to $OUTPUT"
echo "========================================" > "$OUTPUT"
echo "Signal Quality Analysis – $MODE mode" >> "$OUTPUT"
echo "Generated: $(date)" >> "$OUTPUT"
echo "========================================" >> "$OUTPUT"

# Patterns to capture (grep extended regex)
PATTERNS="ENSEMBLE|REGIME|ML PROBS|SIGNAL|EXIT|FILTER|REJECTED|SMC|CONFLUENCE|quality score|ADX|vol_ratio"

# Use grep with context to see surrounding lines (optional: -B1 -A2)
tail -n "$LINES" "$LOG_FILE" | grep -E "$PATTERNS" --color=never >> "$OUTPUT"

echo "" >> "$OUTPUT"
echo "========================================" >> "$OUTPUT"
echo "Top 20 most frequent signal actions (for last $LINES lines):" >> "$OUTPUT"
tail -n "$LINES" "$LOG_FILE" | grep -oE "SIGNAL [A-Z]+/[A-Z]+ → (BUY|SELL|HOLD)" | sort | uniq -c | sort -rn >> "$OUTPUT"

echo "" >> "$OUTPUT"
echo "Top 10 exit reasons (last $LINES lines):" >> "$OUTPUT"
tail -n "$LINES" "$LOG_FILE" | grep -oE "EXIT.*\|\ Reason:.*" | sed 's/.*Reason: //' | sort | uniq -c | sort -rn | head -10 >> "$OUTPUT"

echo "" >> "$OUTPUT"
echo "Latest market regime entries:" >> "$OUTPUT"
tail -n "$LINES" "$LOG_FILE" | grep "REGIME:" | tail -5 >> "$OUTPUT"

echo ""
echo "Log extraction complete. File: $OUTPUT"
echo "You can view it with: cat $OUTPUT"
echo "Or download it via SCP/rsync."
