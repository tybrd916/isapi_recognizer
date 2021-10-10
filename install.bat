@echo off
REM script to install python 3.10 and run the yolov5

:: Check for Python Installation
if not exist %LOCALAPPDATA%\Programs\Python\Python39 goto errorNoPython

git clone https://github.com/ultralytics/yolov5.git
pip install -r requirements.txt
cd yolov5
copy NUL __init.py__
pip install -r requirements.txt
cd ..
python yolo_test.py


:: Once done, exit the batch file -- skips executing the errorNoPython section
goto:eof

:errorNoPython
echo.
echo Error^: Python 3.9 not installed, install from https://www.python.org/ftp/python/3.9.7/python-3.9.7-amd64.exe
