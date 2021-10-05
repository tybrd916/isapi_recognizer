import sys
sys.path.insert(0, './yolov5')
import torch
import requests
from requests.auth import HTTPDigestAuth
from decouple import config
from PIL import Image
import numpy as np

import time
import io
import os
import glob
import tempfile
import datetime
from pathlib import Path

from yolov5.utils.plots import Annotator, colors

import logging
logging.disable()

lastInterestCount=0
class yoloTest:
    def __init__(self):
        # Model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m')  # or yolov5m, yolov5l, yolov5x, custom

        # self.model.conf = 0.7
        self.model.conf = 0.05
        self.maximumSnapshots = 10
        self.snapshotCount = 0
        
        if not glob.glob("snapshots"):
            os.mkdir("snapshots")
        self.lastInterestCount=0
        self.objects_of_interest = {
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

        while self.snapshotCount < self.maximumSnapshots:
            self.yolo_magic()
            time.sleep(1)

    def yolo_magic(self):
        results = self.model(self.download_image())
        # print(f"model results loaded")
        # Results
        resultList = results.tolist()
        interestCount=0
        interestStr=""
        # ct stores current time
        ct = datetime.datetime.now()
        for i, (im, pred) in enumerate(zip(results.imgs, results.pred)):
            # print(pred[:, -1])
            for c in pred[:, -1]:
                if results.names[int(c)] in self.objects_of_interest:
                    interestCount=interestCount+1
                    # print(results.names[int(c)])
            if interestCount > 0:
                annotator = Annotator(im, example=str(results.names))
                for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                    label = f'{results.names[int(cls)]} {conf:.2f}'
                    annotator.box_label(box, label, color=colors(cls))
                #TODO: display current image (annotator.im) as image0,  possibly downscale to a thumbnail size too
                im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
                im.save("latest.jpg")
                if interestCount != self.lastInterestCount:
                    interestStr += f"{ct}: "
                    for c in pred[:, -1].unique():
                        if results.names[int(c)] in self.objects_of_interest:
                            n = (pred[:, -1] == c).sum()  # detections per class
                            interestStr += f"{n} {results.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    print(interestStr)
                    self.snapshotCount = self.snapshotCount+1
                    im.save(f"snapshots/{interestStr}.jpg")
                    # results.display(save=True, save_dir=Path(f"snapshots/{ct}")) #save labeled snapshot by date/timestamp
            self.lastInterestCount=interestCount

    def download_image(self):
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


yt=yoloTest()