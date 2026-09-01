#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
if [ ! -x ".venv/bin/python" ]; then
  echo "Dashboard environment not found. Follow README.md to create it."
  exit 1
fi
.venv/bin/python -m streamlit run app.py
