#!/bin/bash
# Two-proxy chain restart: soul (8319) → cloaked (8318) → Anthropic

echo "[start_both] killing old..."
fuser -k 8319/tcp 2>/dev/null
fuser -k 8318/tcp 2>/dev/null
sleep 1

echo "[start_both] starting cloaked proxy :8318..."
nohup python3 /home/nick/cloaked-proxy.py --port 8318 > /tmp/proxy.log 2>&1 &
sleep 2

echo "[start_both] starting soul proxy :8319..."
nohup python3 /home/nick/soul-proxy.py --port 8319 > /tmp/soul.log 2>&1 &
sleep 2

echo "[start_both] verifying..."
ss -tlnp | grep -E "831[89]" && echo "✓ Both UP" || echo "✗ Some DOWN"