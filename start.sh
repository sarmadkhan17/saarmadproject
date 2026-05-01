#!/bin/bash
# CryptoBot v3 Launcher
BOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "================================="
echo "  🤖 CRYPTOBOT v3 — Demo Trading"
echo "================================="
echo "  1. Spot Trading    (BUY/SELL)"
echo "  2. Futures Trading (LONG/SHORT)"
echo "================================="

if [ -z "$1" ]; then
    read -p "  Select mode (1 or 2): " choice
else
    choice="$1"
fi

if [ "$choice" == "2" ]; then
    MODE="futures"
    PORT=5002
    SCREEN_NAME="cryptobot_v3_futures"
    echo "  🚀 Starting FUTURES bot..."
else
    MODE="spot"
    PORT=5002
    SCREEN_NAME="cryptobot_v3_spot"
    echo "  🚀 Starting SPOT bot..."
fi

# Stop any existing v3 processes only (NOT v2)
pkill -f "cryptobot_v3.*launcher.py" 2>/dev/null || true
pkill -f "cryptobot_v3.*spot_bot.py" 2>/dev/null || true
pkill -f "cryptobot_v3.*futures_bot.py" 2>/dev/null || true
sleep 2

# Make sure directories exist
mkdir -p "$BOT_DIR/logs" "$BOT_DIR/data"

# Start dashboard if dashboard dir exists
if [ -d "$BOT_DIR/dashboard" ]; then
    screen -dmS dashboard_v3 bash -c "cd $BOT_DIR/dashboard && PORT=$PORT python3 server.py"
    sleep 2
fi

# Start bot in screen
screen -dmS "$SCREEN_NAME" bash -c "cd $BOT_DIR/bot && BOT_MODE=$MODE python3 launcher.py"
sleep 2

echo ""
echo "✅ Bot started in $MODE mode!"
screen -ls | grep -E "cryptobot|dashboard"
echo ""
echo "Commands:"
echo "  Watch:     screen -r $SCREEN_NAME (Ctrl+A D to exit)"
echo "  Dashboard: http://localhost:$PORT"
echo "  Telegram:  /status /pnl /trades /agents /health"
echo "  Stop v3:   pkill -f 'cryptobot_v3'"
