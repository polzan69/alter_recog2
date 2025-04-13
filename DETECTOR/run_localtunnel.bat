@echo off
echo Starting LocalTunnel for port 5000...
title LocalTunnel - Port 5000

rem Create logs directory if it doesn't exist
if not exist logs mkdir logs

rem Run LocalTunnel in a loop to auto-restart if it fails
:loop
echo [%date% %time%] Starting/Restarting LocalTunnel...

rem Try npx approach first (most reliable)
echo Trying with npx...
npx localtunnel --port 5000 --print-url > logs\localtunnel.log 2>&1

rem If we get here, LocalTunnel has exited
echo [%date% %time%] LocalTunnel exited, restarting in 5 seconds...
timeout /t 5 /nobreak
goto loop 