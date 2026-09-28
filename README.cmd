@echo off
cd /d "%~dp0"
if not exist openjevx.exe (
  echo OpenJevX: missing openjevx.exe next to README.cmd
  exit /b 1
)
openjevx.exe %*
