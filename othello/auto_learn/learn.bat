cd %~dp0

xcopy ..\Release\othello.exe .\ /Y /I /D
xcopy ..\Release\learning.exe .\learning\ /Y /I /D

@rem xcopy ..\othello.net .\ /Y /I /D
@rem xcopy ..\learning\teacher.log .\learning\ /Y /I /D

@setlocal enabledelayedexpansion
for /l %%n in (0,1,10) do (
@echo start !n! %date% %time%
for /l %%i in (0,1,10) do (
othello.exe learn
)

pushd learning
learning.exe
popd
@echo end !n! %date% %time%
)


pause
