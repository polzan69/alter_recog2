@echo off
echo Checking LocalTunnel status...

set LOG_FILE=logs\localtunnel.log

rem Check if log file exists
if not exist %LOG_FILE% (
    echo LocalTunnel log file not found.
    echo Run direct_localtunnel.bat to create a new tunnel.
    goto end
)

rem Read the URL from the log file
set /p URL=<%LOG_FILE%

rem Display the URL
echo Found URL: %URL%

rem Check if URL is reachable
echo Testing connection to URL...
curl -s -I -m 5 %URL% > nul
if %ERRORLEVEL% EQU 0 (
    echo The tunnel is ACTIVE and working.
) else (
    echo The tunnel appears to be DOWN or unreachable.
    echo Run direct_localtunnel.bat to create a new tunnel.
)

:end
pause 