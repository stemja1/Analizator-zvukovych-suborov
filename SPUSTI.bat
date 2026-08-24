@echo off
rem ============================================================
rem  SPUSTI.BAT — spusti Analyzator zvukovych suborov
rem  Pouzitie: dvakrat klikni na tento subor.
rem ============================================================
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Prvy krat: pripravujem prostredie ^(trva par minut, len raz^)...
    python -m venv .venv
    call .venv\Scripts\activate
    pip install -r requirements.txt
)
call .venv\Scripts\activate
python main_gui.py
if errorlevel 1 pause
