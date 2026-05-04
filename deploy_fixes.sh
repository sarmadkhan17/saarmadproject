#!/bin/bash
# Deploy fixes to /root/cryptobot_v3
DEST=/root/cryptobot_v3

echo "Creating directories..."
mkdir -p $DEST/data $DEST/logs $DEST/bot $DEST/dashboard

echo "Copying fixed files..."
cp /home/sarmad/cryptobot_v3/bot/env_config.py    $DEST/bot/env_config.py
cp /home/sarmad/cryptobot_v3/bot/base_bot.py      $DEST/bot/base_bot.py
cp /home/sarmad/cryptobot_v3/bot/data_feed.py     $DEST/bot/data_feed.py
cp /home/sarmad/cryptobot_v3/bot/ai_strategy.py   $DEST/bot/ai_strategy.py
cp /home/sarmad/cryptobot_v3/dashboard/server.py  $DEST/dashboard/server.py

echo "Restarting services..."
pkill -f 'cryptobot_v3' 2>/dev/null
sleep 2
cd $DEST && bash start.sh 2

echo "Done! Dashboard: http://localhost:5002"
