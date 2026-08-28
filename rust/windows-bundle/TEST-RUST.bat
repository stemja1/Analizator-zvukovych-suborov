@echo off
setlocal
rem ============================================
rem  Analyzator zvukovych suborov - RUST TEST
rem  (rovnake vysledky ako Python verzia)
rem ============================================
chcp 65001 >nul
cd /d "%~dp0"

set "EXE=%~dp0analyzator-rs.exe"
set "POPISY=%~dp0popisy.txt"
set "MODELDIR="

if not exist "%EXE%" (
  echo CHYBA: Nenasiel som analyzator-rs.exe v tomto priecinku.
  echo Pravdepodobne ho zmazal antivirus Windows Defender.
  echo Pozri README-RUST.txt, kapitolu o antivirusovej kontrole.
  pause
  exit /b 1
)

rem -- 1) priecinok je umiestneny V hlavnom priecinku programu
rem      (odporucane): model aj naucene data su o uroven vyssie
if exist "%~dp0..\models\clap_htsat_unfused_onnx\clap_audio.onnx" (
  cd /d "%~dp0.."
  goto run
)

rem -- 2) model je v priecinku "model" priamo pri programe
if exist "%~dp0model\clap_audio.onnx" (
  set "MODELDIR=--model-dir "%~dp0model""
  goto run
)

rem -- 3) model je v aktualnom adresari
if exist "models\clap_htsat_unfused_onnx\clap_audio.onnx" goto run

echo POZOR: Nenasiel som AI model (clap_audio.onnx).
echo Nakopiruj tento priecinok do hlavneho priecinka programu
echo (tam, kde je SPUSTI.bat a priecinok "models") a spusti znovu.
echo.

:run
set "PRIECINOK=%~1"
if "%PRIECINOK%"=="" (
  echo Zadaj cestu k priecinku so zvukovymi subormi a stlac Enter
  echo (napriklad C:\Users\TvojeMeno\Desktop\Zvuky):
  set /p PRIECINOK="> "
)
if "%PRIECINOK%"=="" (echo Nezadana cesta - koniec. & pause & exit /b 1)
if not exist "%PRIECINOK%" (echo Priecinok neexistuje: %PRIECINOK% & pause & exit /b 1)

echo.
echo Analyzujem: %PRIECINOK%
echo (Mozes tahat priecinok aj myskou na TEST-RUST.bat)
echo.
"%EXE%" "%PRIECINOK%" --popisy "%POPISY%" %MODELDIR%
echo.
pause
