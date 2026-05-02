#!/bin/bash
# Deploy CryptoBot v3 to Contabo via Docker
# Usage: bash deploy_contabo.sh

set -e

SERVER="root@185.249.225.94"
REMOTE_DIR="/root/cryptobot_v3"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "========================================"
echo "  CryptoBot v3 → Contabo (Docker)"
echo "========================================"

# ── Step 1: Install Docker on remote if missing ───────────────────
echo ""
echo "[1/5] Checking Docker on remote..."
ssh "$SERVER" bash << 'REMOTE_DOCKER'
if ! command -v docker &>/dev/null; then
    echo "  → Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
else
    echo "  → Docker already installed: $(docker --version)"
fi

if ! docker compose version &>/dev/null 2>&1; then
    echo "  → Installing docker-compose plugin..."
    apt-get install -y docker-compose-plugin
fi
echo "  ✓ Docker ready"
REMOTE_DOCKER

# ── Step 2: Transfer code ─────────────────────────────────────────
echo ""
echo "[2/5] Syncing code..."
rsync -az --progress \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.env' \
    --exclude='venv/' \
    --exclude='.venv/' \
    --exclude='logs/' \
    --exclude='.git/' \
    --exclude='.claude/' \
    --exclude='install.cmd' \
    "$LOCAL_DIR/" "$SERVER:$REMOTE_DIR/"
echo "  ✓ Code synced"

# ── Step 3: Transfer .env ─────────────────────────────────────────
echo ""
echo "[3/5] Transferring .env..."
rsync -az "$LOCAL_DIR/.env" "$SERVER:$REMOTE_DIR/.env"
ssh "$SERVER" "chmod 600 $REMOTE_DIR/.env"
echo "  ✓ .env secured"

# ── Step 4: Transfer trained models + state ───────────────────────
echo ""
echo "[4/5] Syncing models and state..."
rsync -az --progress \
    --include='*.pkl' \
    --include='*.pt' \
    --include='*.keras' \
    --include='*.json' \
    --include='futures/' \
    --include='futures/*.pkl' \
    --include='futures/*.json' \
    --exclude='*.log' \
    "$LOCAL_DIR/data/" "$SERVER:$REMOTE_DIR/data/"
echo "  ✓ Models and state synced"

# ── Step 5: Build image and start containers ──────────────────────
echo ""
echo "[5/5] Building image and starting containers on remote..."
ssh "$SERVER" bash << REMOTE_RUN
set -e
cd $REMOTE_DIR

echo "  → Building Docker image (first build takes ~5 min)..."
docker compose build

echo "  → Starting containers..."
docker compose up -d

echo ""
echo "  Container status:"
docker compose ps
REMOTE_RUN

echo ""
echo "========================================"
echo "  Deployment complete!"
echo "========================================"
echo ""
echo "Useful commands (run via SSH):"
echo "  ssh $SERVER"
echo "  cd $REMOTE_DIR"
echo ""
echo "  Logs:      docker compose logs -f spot-bot"
echo "             docker compose logs -f futures-bot"
echo "             docker compose logs -f dashboard"
echo ""
echo "  Restart:   docker compose restart spot-bot"
echo "             docker compose restart futures-bot"
echo ""
echo "  Stop all:  docker compose down"
echo "  Start all: docker compose up -d"
echo ""
echo "  Dashboard: http://185.249.225.94:5002"
echo ""
echo "NOTE: Telegram /restart commands won't work in Docker."
echo "      Use 'docker compose restart <service>' instead."
