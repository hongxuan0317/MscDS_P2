@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Dashboard environment not found. Follow README.md to create it.
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py
