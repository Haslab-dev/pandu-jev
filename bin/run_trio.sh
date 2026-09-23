#!/usr/bin/env bash
set -e

# Trio Snake Arena: Pandu-Jev vs Laya-CoreML vs Jev (TypeSafe System One, Max Speed)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINI_JEV_DIR="$(dirname "$SCRIPT_DIR")"
LAYA_DIR="/Users/hy4-mac-002/hasdev/research/laya-coreml"

echo "Launching 3-Way Snake Arena (Pandu vs Laya vs Jev) on macOS..."

osascript << EOF
tell application "Terminal"
    -- 1. Window 1 (Left): Pandu-Jev NLP (~19.3M, Local MPS)
    set tab1 to (do script "cd $MINI_JEV_DIR && .venv/bin/python bin/run_snake_ui.py")
    set win1 to first window whose tabs contains tab1
    set number of columns of win1 to 105
    set number of rows of win1 to 36
    set position of win1 to {0, 25}

    delay 0.5

    -- 2. Window 2 (Right): Laya-CoreML (~164M, Local ANE)
    set tab2 to (do script "cd $LAYA_DIR && /Users/hy4-mac-002/.pyenv/shims/laya-coreml-snake --model ./models/snake")
    set win2 to first window whose tabs contains tab2
    set number of columns of win2 to 105
    set number of rows of win2 to 36
    set position of win2 to {720, 25}

    delay 0.5

    -- 3. Window 3 (Center / Front): TypeSafe Jev (Cloud System One, MAX SPEED)
    set tab3 to (do script "cd $MINI_JEV_DIR && .venv/bin/python bin/run_snake_ui.py --jev --max-speed")
    set win3 to first window whose tabs contains tab3
    set number of columns of win3 to 105
    set number of rows of win3 to 36
    set position of win3 to {360, 140}

    activate
end tell

tell application "System Events"
    set frontmost of process "Terminal" to true
end tell
EOF

echo "All 3 Snake models are running live!"
