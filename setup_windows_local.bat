@echo off
echo Running batch file: %~f0
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM ---------------------------------------------------------------------------
REM Local-only setup: installs the on-disk ondil / VineCopulas forks without
REM touching git (no submodule update / checkout / pull). Use this when you have
REM copied the folder to a machine without git access. The original
REM setup_windows.bat (git-based) is left untouched.
REM
REM The on-disk forks are already on the correct branches
REM (ondil -> bivariate_copula, VineCopulas -> dev), so no checkout is needed.
REM ---------------------------------------------------------------------------

REM ---- We assume you run this from Anaconda Prompt (base env active) ----
echo [*] Using Anaconda base environment
call conda activate base

REM ---- Create env only if it does not exist yet ----
echo [*] Checking if env 'online_copula_exp' exists...
conda env list | findstr /b /c:"online_copula_exp " >nul
if errorlevel 1 (
    echo [*] Creating env 'online_copula_exp' with Python 3.12...
    call conda create -y -n online_copula_exp python=3.12 || exit /b
) else (
    echo [*] Env 'online_copula_exp' already exists, reusing it.
)

REM ---- Activate env ----
echo [*] Activating env 'online_copula_exp'...
call conda activate online_copula_exp || exit /b

REM ---- Install Python deps from req.txt ----
echo [*] Installing Python requirements from req.txt...
python -m pip install -r req.txt || exit /b

REM ---- Install ondil (editable) from the on-disk folder ----
echo [*] Installing ondil (editable, from disk)...
cd /d "%~dp0"
cd ondil || exit /b
python -m pip install -e . || exit /b
cd /d "%~dp0"

REM ---- Install VineCopulas (editable) from the on-disk folder ----
echo [*] Installing VineCopulas (editable, from disk)...
cd VineCopulas || exit /b
python -m pip install -e . || exit /b
cd /d "%~dp0"

echo.
echo [OK] Local setup finished. Environment 'online_copula_exp' is ready.
echo      Interpreter: %USERPROFILE%\conda3\envs\online_copula_exp\python.exe
echo.
pause
