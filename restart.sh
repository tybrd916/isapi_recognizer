#/bin/bash

cd /home/tcarr/isapi_recognizer
#Kill any running yolo before restarting
ps -ef|grep python|grep yolo_test.py|grep -v grep|tr -s " "|cut -d " " -f 2|xargs -I {} kill -9 {};

#Start new Yolo session
python3 yolo_test.py
