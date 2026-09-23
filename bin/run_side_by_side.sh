#!/usr/bin/env bash
set -e

# Side-by-Side Dual Snake Arena: Pandu-Jev NLP (~19.3M) vs Laya CoreML (~164M)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINI_JEV_DIR="$(dirname "$SCRIPT_DIR")"
LAYA_DIR="/Users/hy4-mac-002/hasdev/research/laya-coreml"

echo "Launching Side-by-Side Snake Arena on macOS..."

osascript << EOF
tell application "Terminal"
    -- 1. Left Window: Pandu-Jev NLP (19.3M, MPS FP16)
    set tab1 to (do script "cd $MINI_JEV_DIR && .venv/bin/python bin/run_snake_ui.py")
    set win1 to first window whose tabs contains tab1
    set number of columns of win1 to 105
    set number of rows of win1 to 36
    set position of win1 to {0, 25}

    delay 0.5

    -- 2. Right Window: Laya CoreML (164M, ANE FP16)
    set tab2 to (do script "cd $LAYA_DIR && /Users/hy4-mac-002/.pyenv/shims/laya-coreml-snake --model ./models/snake")
    set win2 to first window whose tabs contains tab2
    set number of columns of win2 to 105
    set number of rows of win2 to 36
    set position of win2 to {720, 25}

    activate
end tell

tell application "System Events"
    set frontmost of process "Terminal" to true
end tell
EOF

echo "Both windows launched side-by-side!"
