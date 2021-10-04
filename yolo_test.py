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

last_image_len = 0

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom
model.conf = 0.7

# Clear output pictures
def clear_output_dir():
    for hgx in glob.glob("runs/detect/exp*/image*jpg"):
        os.remove(hgx)
    for hgx in glob.glob("runs/detect/exp*"):
        os.rmdir(hgx)
    os.rmdir('runs/detect')
    os.rmdir('runs')

def yolo_magic():
    results = model(download_image())
    print(f"model results loaded")
    # Results
    results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
    results.save()  # or .show(), .save(), .crop(), .pandas(), etc.

def download_image():
    hostaddress=config('ISAPI_HOST')
    print(f'downloading http://{hostaddress}/ISAPI/Streaming/channels/101/picture')
    # imageResult = requests.get(f'http://{hostaddress}/ISAPI/Streaming/channels/101/picture', auth=HTTPDigestAuth(config('ISAPI_USER'), config('ISAPI_PASSWORD')))
    # return Image.fromarray(np.asarray(imageResult.content))
    # return Image.frombytes(imageResult.content)
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
    buffer.close()
    return i

clear_output_dir() #TODO: make conditional
for i in range(10):
    time.sleep(2)
    yolo_magic()
# download_image()