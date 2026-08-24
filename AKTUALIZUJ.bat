@echo off
rem ============================================================
rem  AKTUALIZUJ.BAT - stiahne najnovsiu verziu aplikacie z GitHubu
rem  Pouzitie: dvakrat klikni na tento subor. Hotovo.
rem  Funguje aj BEZ nainstalovaneho gitu: stiahne ZIP a rozbali ho.
rem ============================================================
cd /d "%~dp0"

rem -- ak existuje odlozena novsia verzia tohto suboru, nainstaluje sa
rem    a subor sa spusti znovu (bezpecne prepisanie sameho seba)
if exist "%~dp0AKTUALIZUJ.bat.new" (
    move /Y "%~dp0AKTUALIZUJ.bat.new" "%~dp0AKTUALIZUJ.bat" >nul
    start "" "%~dp0AKTUALIZUJ.bat"
    exit /b 0
)

echo Zistujem aktualizacie z GitHubu...

where git >nul 2>&1
if not errorlevel 1 (
    if exist ".git" (
        echo Git najdeny - stahujem klasicky (git pull)...
        git pull
        goto hotovo
    )
)

echo Git nie je nainstalovany - pouzijem bezgitovy rezim (ZIP)...
set "ZIPURL=https://github.com/stemja1/Analizator-zvukovych-suborov/archive/refs/heads/main.zip"
set "ZMENA=%TEMP%\analyzator_aktualizacia"

echo Stahujem aktualizaciu...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%ZIPURL%' -OutFile '%ZMENA%.zip'"
if errorlevel 1 goto chyba_stahovania

echo Rozbalujem...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Remove-Item -Recurse -Force '%ZMENA%' -ErrorAction SilentlyContinue; Expand-Archive -Path '%ZMENA%.zip' -DestinationPath '%ZMENA%' -Force"
if errorlevel 1 goto chyba_stahovania

set "SRC=%ZMENA%\Analizator-zvukovych-suborov-main"
if not exist "%SRC%" goto chyba_stahovania

rem -- nakopiruj nove subory (nemeni sa .venv ani stiahnuty model)
robocopy "%SRC%" "%~dp0" /E /XD .venv models __pycache__ .git /XF AKTUALIZUJ.bat /NFL /NDL /NJH /NJS >nul
if errorlevel 8 goto chyba_kopirovania

rem -- tento .bat sa nemoze prepisat pocas svojho behu,
rem    nova verzia sa odlozi a nainstaluje pri dalsom spusteni
copy /Y "%SRC%\AKTUALIZUJ.bat" "%~dp0AKTUALIZUJ.bat.new" >nul

:hotovo
rem -- pripadne nove kniznice (rychla kontrola, nic sa nestane ak su aktualne)
if exist ".venv\Scripts\python.exe" (
    echo Kontrolujem kniznice...
    ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt
)
echo.
echo ================================================
echo  Hotove! Mas najnovsiu verziu aplikacie.
echo  Spusti SPUSTI.BAT
echo ================================================
pause
exit /b 0

:chyba_stahovania
echo.
echo NEPODARILO SA STIAHNUT AKTUALIZACIU.
echo Skontroluj pripojenie k internetu a skus este raz.
pause
exit /b 1

:chyba_kopirovania
echo.
echo NEPODARILO SA NAKOPIROVAT NOVE SUBORY.
pause
exit /b 1
