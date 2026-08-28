@echo off
rem ============================================================
rem  DIAGNOZA.BAT - zisti, co sa pokazilo v Analyzatore
rem  Dvakrat klikni na tento subor. Otvori sa poznamkovy blok
rem  s vysledkom - skopiruj CELY text a posli ho spat.
rem ============================================================
cd /d "%~dp0"
set "OUT=%~dp0DIAGNOZA-VYSLEDOK.txt"
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo DIAGNOZA ANALYZATORA - %date% %time% > "%OUT%"
echo Priecinok: %~dp0 >> "%OUT%"

echo. >> "%OUT%"
echo === 1. SUBORY PROGRAMU === >> "%OUT%"
dir /b "%~dp0" >> "%OUT%" 2>&1

echo. >> "%OUT%"
echo === 2. AI MODEL (models\clap_htsat_unfused_onnx) === >> "%OUT%"
dir "%~dp0models\clap_htsat_unfused_onnx" >> "%OUT%" 2>&1

echo. >> "%OUT%"
echo === 3. PYTHON PROSTREDIE (.venv) === >> "%OUT%"
if exist "%~dp0.venv\Scripts\python.exe" (echo .venv najdeny OK >> "%OUT%") else (echo CHYBA: .venv NEexistuje >> "%OUT%")
"%PY%" --version >> "%OUT%" 2>&1

echo. >> "%OUT%"
echo === 4. AI KNIZNICE - najdolezitejsi test === >> "%OUT%"
"%PY%" -c "import onnxruntime; print('onnxruntime OK', onnxruntime.__version__, onnxruntime.get_available_providers())" >> "%OUT%" 2>&1
"%PY%" -c "import numpy; print('numpy OK', numpy.__version__)" >> "%OUT%" 2>&1
"%PY%" -c "import soundfile; print('soundfile OK')" >> "%OUT%" 2>&1
"%PY%" -c "import transformers; print('transformers OK')" >> "%OUT%" 2>&1
"%PY%" -c "import PyQt6.QtWidgets; print('PyQt6 OK')" >> "%OUT%" 2>&1

echo. >> "%OUT%"
echo === 5. DLL KNIZNICE V PROSTREDI (ci ich nieco nezmazalo) === >> "%OUT%"
dir /b "%~dp0.venv\Lib\site-packages\onnxruntime\capi\*.dll" >> "%OUT%" 2>&1

echo. >> "%OUT%"
echo === 6. RUST PRIECINOK === >> "%OUT%"
dir /b "%~dp0analyzator-rs-windows" >> "%OUT%" 2>&1

echo. >> "%OUT%"
echo === 7. WINDOWS DEFENDER - CO ZABLOKOVAL === >> "%OUT%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-MpThreatDetection | Select-Object -First 10 InitialDetectionTime, Resources | Format-List" >> "%OUT%" 2>&1

echo. >> "%OUT%"
echo === HOTOVO - skopiruj vsetko a posli === >> "%OUT%"
start notepad "%OUT%"
exit /b 0
