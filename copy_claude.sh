#!/bin/bash
set -e
SERVER="root@185.249.225.94"

echo "[1/4] Creating remote directories..."
ssh $SERVER "mkdir -p ~/.local/share/claude/versions/2.1.122 ~/.local/bin ~/.claude/commands"

echo "[2/4] Copying Claude Code binary..."
rsync -az ~/.local/share/claude/versions/2.1.122 $SERVER:~/.local/share/claude/versions/
ssh $SERVER "ln -sf /root/.local/share/claude/versions/2.1.122 /root/.local/bin/claude && chmod +x /root/.local/bin/claude"

echo "[3/4] Copying settings and skills..."
rsync -az ~/.claude/settings.json $SERVER:~/.claude/settings.json
rsync -az ~/.claude/commands/ $SERVER:~/.claude/commands/

echo "[4/4] Copying CLAUDE.md..."
rsync -az ~/cryptobot_v3/bot/CLAUDE.md $SERVER:~/cryptobot_v3/bot/CLAUDE.md

echo ""
echo "Adding claude to PATH on remote..."
ssh $SERVER "grep -q '.local/bin' ~/.bashrc || echo 'export PATH=\$PATH:\$HOME/.local/bin' >> ~/.bashrc"

echo ""
echo "Verifying..."
ssh $SERVER "/root/.local/bin/claude --version"
echo "All done!"
