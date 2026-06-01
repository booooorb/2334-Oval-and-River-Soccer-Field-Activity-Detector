@echo off
cd /d "%~dp0"
"D:\Downloads\a5_code_data\python-embed\python.exe" src\label_server.py --host 127.0.0.1 --port 8123
pause
