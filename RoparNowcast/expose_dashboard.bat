@echo off
echo 🌩️ Starting Cloudflare Tunnel...
echo    Make sure your Dashboard is running!
echo    Targeting: http://localhost:8502
echo.
cloudflared tunnel --url http://localhost:8502
pause
