import torch
import requests
from requests.auth import HTTPDigestAuth
from decouple import config
from PIL import Image
# import numpy as np

import time
import io
import os
import glob
import tempfile

import json

last_image_len = 0

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')  # or yolov5m, yolov5l, yolov5x, custom
model.conf = 0.7
# model.conf = 0.05
objects_of_interest = {
    "person": '',
    "bicycle": '',
    "car": '',
    "motorcycle": '',
    "bus": '',
    "train": '',
    "truck": '',
    "bird": '',
    "cat": '',
    "dog": '',
    "horse": '',
    "sheep": '',
    "cow": '',
    "elephant": '',
    "bear": '',
    "zebra": '',
    "giraffe": '',
}

def yolo_magic():
    results = model(download_image())
    # print(f"model results loaded")
    # Results
    resultList = results.tolist()
    interestCount=0
    for det in resultList:
        # print(det.pred[:, -1])
        for c in det.pred[:, -1]:
            if results.names[int(c)] in objects_of_interest:
                interestCount=interestCount+1
                print(results.names[int(c)])
    print(f"# Objects: {interestCount}")
    results.display(save=True)

def download_image():
    hostaddress=config('ISAPI_HOST')
    # print(f'downloading http://{hostaddress}/ISAPI/Streaming/channels/101/picture')
    buffer = tempfile.SpooledTemporaryFile(max_size=1e9)
    r = requests.get(f'http://{hostaddress}/ISAPI/Streaming/channels/101/picture', auth=HTTPDigestAuth(config('ISAPI_USER'), config('ISAPI_PASSWORD')), stream=True)
    if r.status_code == 200:
        downloaded = 0
        filesize = int(r.headers['content-length'])
        for chunk in r.iter_content(chunk_size=1024):
            downloaded += len(chunk)
            buffer.write(chunk)
            # print(downloaded/filesize)
        buffer.seek(0)
        i = Image.open(io.BytesIO(buffer.read()))
        # i.save(os.path.join(out_dir, 'image.jpg'), quality=85)
    else:
        print(r)
    buffer.close()
    return i

while True:
    yolo_magic()
    time.sleep(1)
# download_image()