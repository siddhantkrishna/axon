#!/bin/bash
# AXON one-click launcher for macOS
set -e
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null 2>&1; then
  echo 'Python 3 is required. Install it from https://www.python.org/downloads/'
  read -r -p 'Press Return to close...'
  exit 1
fi
if [ ! -d .venv ]; then
  echo 'Creating local virtual environment...'
  python3 -m venv .venv
fi
source .venv/bin/activate
if ! python -c 'import numpy, sklearn, PIL, matplotlib' 2>/dev/null; then
  echo 'Installing the project dependencies...'
  python -m pip install -r requirements.txt
fi
if ! python -c 'import tkinter' 2>/dev/null; then
  echo 'Tkinter is missing from this Python installation.'
  echo 'Please use Python from python.org, or install matching Homebrew python-tk.'
  echo 'See README.md. Training still works from Terminal.'
  read -r -p 'Press Return to close...'
  exit 1
fi
python -m axon demo
