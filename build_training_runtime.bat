@echo off
setlocal

set SCRIPT_DIR=%~dp0
set PYTHON_EXE=%LocalAppData%\Programs\Python\Python312\python.exe

if not exist "%PYTHON_EXE%" (
  set PYTHON_EXE=python
)

if "%~1"=="" (
  set TARGET_DIR=%SCRIPT_DIR%dist\GeckoAI
) else (
  set TARGET_DIR=%~1
)

echo ================================================================
echo Building training runtime into:
echo   %TARGET_DIR%
echo ================================================================

"%PYTHON_EXE%" "%SCRIPT_DIR%src\ai_labeller\auto_build_training_runtime.py" "%TARGET_DIR%"
set EXIT_CODE=%ERRORLEVEL%

echo.
if %EXIT_CODE% EQU 0 (
  echo Runtime build completed successfully.
) else (
  echo Runtime build failed with code %EXIT_CODE%.
)

pause
exit /b %EXIT_CODE%
