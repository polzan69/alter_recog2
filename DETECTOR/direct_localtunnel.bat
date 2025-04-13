@echo off
echo Starting LocalTunnel for port 5000...
echo This window must remain open for the tunnel to work.
echo.

rem Create logs directory if it doesn't exist
if not exist logs mkdir logs

echo [%date% %time%] Starting LocalTunnel...
echo The URL will appear below once connected:
echo.

rem Run LocalTunnel directly without a loop
npx localtunnel@latest --port 5000 