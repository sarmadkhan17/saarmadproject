#!/bin/bash
# Migration script: local → Contabo VPS
# Usage: bash migrate_to_contabo.sh

set -e
SERVER="root@185.249.225.94"
REMOTE_DIR="/root/cryptobot_v3"
LOCAL_DIR="$HOME/cryptobot_v3"

echo ""
echo "========================================"
echo "  CryptoBot v3 → Contabo Migration"
echo "========================================"
echo ""

# ── Step 1: Remote system setup ──────────────────────────────────
echo "[1/5] Setting up remote server..."
ssh "$SERVER" bash << 'REMOTE_SETUP'
set -e
export DEBIAN_FRONTEND=noninteractive

echo "  → Updating apt..."
apt-get update -qq

echo "  → Installing system packages..."
apt-get install -y -qq python3 python3-pip python3-venv python3-full screen rsync git curl 2>/dev/null

echo "  → Creating virtualenv at /root/cryptobot_v3/venv ..."
python3 -m venv /root/cryptobot_v3/venv

echo "  → Installing Python bot dependencies into venv..."
/root/cryptobot_v3/venv/bin/pip install -q --upgrade pip
/root/cryptobot_v3/venv/bin/pip install -q \
    numpy pandas requests pyyaml python-dotenv \
    scikit-learn lightgbm joblib scipy \
    groq ccxt ta

echo "  → Installing TensorFlow + Keras (this may take a few minutes)..."
/root/cryptobot_v3/venv/bin/pip install -q tensorflow keras

echo "  → Creating directories..."
mkdir -p /root/cryptobot_v3/data
mkdir -p /root/cryptobot_v3/logs
mkdir -p /root/cryptobot_v3/bot

echo "  ✓ Remote server ready"
REMOTE_SETUP

# ── Step 2: Transfer bot code ─────────────────────────────────────
echo ""
echo "[2/5] Transferring bot code..."
rsync -az --progress \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.env' \
    "$LOCAL_DIR/bot/" "$SERVER:$REMOTE_DIR/bot/"
echo "  ✓ Bot code transferred"

# ── Step 3: Transfer config and launcher ─────────────────────────
echo ""
echo "[3/5] Transferring config files..."
rsync -az \
    "$LOCAL_DIR/config_spot.yaml" \
    "$LOCAL_DIR/config_futures.yaml" \
    "$LOCAL_DIR/start.sh" \
    "$SERVER:$REMOTE_DIR/"
echo "  ✓ Config files transferred"

# ── Step 4: Transfer .env (API keys) ─────────────────────────────
echo ""
echo "[4/5] Transferring .env (API keys)..."
rsync -az "$LOCAL_DIR/.env" "$SERVER:$REMOTE_DIR/.env"
ssh "$SERVER" "chmod 600 $REMOTE_DIR/.env"
echo "  ✓ .env transferred and secured"

# ── Step 5: Transfer models and state (optional) ─────────────────
echo ""
echo "[5/5] Transferring trained models and state..."
rsync -az --progress \
    --include='*.pkl' \
    --include='*.keras' \
    --include='*.json' \
    --exclude='*' \
    "$LOCAL_DIR/data/" "$SERVER:$REMOTE_DIR/data/"
echo "  ✓ Models and state transferred"

# ── Final: Set permissions and verify ────────────────────────────
echo ""
echo "Finalizing..."
ssh "$SERVER" bash << REMOTE_FINAL
chmod +x $REMOTE_DIR/start.sh

# Patch start.sh on remote to use venv python
sed -i 's|python3 launcher.py|/root/cryptobot_v3/venv/bin/python3 launcher.py|g' $REMOTE_DIR/start.sh
sed -i 's|python3 server.py|/root/cryptobot_v3/venv/bin/python3 server.py|g' $REMOTE_DIR/start.sh

echo ""
echo "========================================"
echo "  Migration complete!"
echo "========================================"
echo ""
echo "To start the bot on Contabo, SSH in and run:"
echo "  ssh root@185.249.225.94"
echo "  bash ~/cryptobot_v3/start.sh 2    # futures"
echo "  bash ~/cryptobot_v3/start.sh 1    # spot"
echo ""
echo "To watch logs:"
echo "  screen -r cryptobot_v3_futures"
echo "  tail -f ~/cryptobot_v3/logs/futures_bot.log"
echo ""
echo "Remote Python version:"
python3 --version
echo "Remote pip packages (key ones):"
pip3 show tensorflow keras lightgbm scikit-learn 2>/dev/null | grep -E "Name:|Version:"
REMOTE_FINAL
