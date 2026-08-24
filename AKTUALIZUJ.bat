@echo off
rem ============================================================
rem  AKTUALIZUJ.BAT — stiahne najnovsiu verziu aplikacie z GitHubu
rem  Pouzitie: dvakrat klikni na tento subor. Hotovo.
rem ============================================================
cd /d "%~dp0"
echo Zistujem aktualizacie z GitHubu...
git pull
echo.
echo ================================================
echo  Hotove. Ak hore pisalo "Already up to date.",
echo  mas uplne najnovsiu verziu. Inak sa stiahli
echo  nove subory - mozes spustit SPUSTI.BAT
echo ================================================
pause
