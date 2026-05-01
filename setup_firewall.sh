#!/bin/bash
set -e

echo "Setting up firewall..."
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp
ufw allow 5002/tcp
ufw --force enable
ufw status
echo "Firewall active!"
