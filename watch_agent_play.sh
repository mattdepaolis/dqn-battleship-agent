#!/bin/bash
# Quick script to watch the DQN agent play Battleship

echo "🎮 DQN Agent Playing Battleship"
echo ""
echo "Options:"
echo "  1) Auto-play (1 second delay)"
echo "  2) Fast playback (0.3 seconds)"  
echo "  3) Interactive (press Enter each move)"
echo "  4) Analysis mode (move 10)"
echo ""
read -p "Choose option (1-4) [1]: " choice
choice=${choice:-1}

cd "$(dirname "$0")"

case $choice in
    1)
        echo "▶️  Auto-play mode (1s delay)..."
        uv run python rl/visualize_play.py --delay 1.0 --seed 42
        ;;
    2)
        echo "⚡ Fast playback (0.3s)..."
        uv run python rl/visualize_play.py --delay 0.3 --seed 42
        ;;
    3)
        echo "⏸️  Interactive mode (press Enter)..."
        uv run python rl/visualize_play.py --interactive --seed 42
        ;;
    4)
        echo "🔍 Analyzing move 10..."
        uv run python rl/visualize_play.py --analyze-move 10 --seed 42
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac
