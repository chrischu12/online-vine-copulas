@echo off
echo Running batch file: %~f0
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM ---- We assume you run this from Anaconda Prompt ----
REM base env is already active there, so conda is on PATH
REM this instally my local ondil and vinecopulas edits in editable mode

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

REM ---- Make sure the ondil submodule is initialized ----
echo [*] Initializing/updating submodule 'ondil'...
git submodule update --init --recursive ondil || exit /b

echo [*] Initializing/updating submodule 'VineCopulas'...
git submodule update --init --recursive VineCopulas || exit /b

REM ---- Install Python deps from req.txt ----
echo [*] Installing Python requirements from req.txt...
python -m pip install -r req.txt || exit /b

REM ---- Install ondil (editable) on branch bivariate_copula ----
echo [*] Preparing ondil (branch bivariate_copula)...
cd ondil || exit /b
git fetch || exit /b
git checkout bivariate_copula || git switch bivariate_copula || exit /b
git pull || exit /b

echo [*] Installing ondil (editable)...
REM for installing the submodule 
cd /d "%~dp0"
cd ondil || exit /b
python -m pip install -e . || exit /b
cd .. || exit /b

REM ---- Install VineCopulas (editable) ----
echo [*] Installing VineCopulas (editable)...
cd VineCopulas || exit /b
git fetch || exit /b
git checkout dev || git switch dev || exit /b
git pull || exit /b

echo [*] Installing VineCopulas (editable)...
REM for installing the submodule 
cd /d "%~dp0"
cd VineCopulas || exit /b
python -m pip install -e . || exit /b
cd .. || exit /b

echo.
echo [✓] Setup finished. Environment 'online_copula_exp' is ready.
echo     In Spyder, select this interpreter:
echo     %USERPROFILE%\conda3\envs\online_copula_exp\python.exe
echo.
pause

