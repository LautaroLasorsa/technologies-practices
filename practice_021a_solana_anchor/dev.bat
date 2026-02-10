@echo off
setlocal

if "%1"=="" goto :usage

if "%1"=="up"        docker compose up -d --build & goto :eof
if "%1"=="down"      docker compose down & goto :eof
if "%1"=="shell"     docker compose exec dev bash & goto :eof
if "%1"=="build"     docker compose exec dev anchor build & goto :eof
if "%1"=="test"      docker compose exec dev anchor test & goto :eof
if "%1"=="init"      docker compose exec dev anchor init %2 & goto :eof
if "%1"=="validator" docker compose exec dev solana-test-validator & goto :eof
if "%1"=="logs"      docker compose logs -f & goto :eof
if "%1"=="versions"  docker compose exec dev bash -c "rustc --version && solana --version && anchor --version && node --version" & goto :eof

echo Unknown command: %1
goto :usage

:usage
echo Usage: dev.bat ^<command^> [args]
echo.
echo Commands:
echo   up          Build image and start container in background
echo   down        Stop and remove container
echo   shell       Open bash shell inside the container
echo   build       Run 'anchor build' inside the container
echo   test        Run 'anchor test' inside the container
echo   init ^<name^> Run 'anchor init ^<name^>' inside the container
echo   validator   Start solana-test-validator inside the container
echo   logs        Follow container logs
echo   versions    Print installed tool versions
goto :eof
