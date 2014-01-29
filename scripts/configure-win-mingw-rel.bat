@echo off

set SCRIPTS_DIR=%CD%
set BUILD_DIR="build_mingw_rel"

mkdir ..\%BUILD_DIR%
cd ..\%BUILD_DIR%
cmake .. -G"MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cd %SCRIPTS_DIR%
