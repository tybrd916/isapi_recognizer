import torch
import requests
from requests.auth import HTTPDigestAuth
from decouple import config

def yolo_magic():
    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom

    # # Images
    img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

    # # Inference
    results = model(img)

    # Results
    results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
    # results.show()  # or .show(), .save(), .crop(), .pandas(), etc.

def download_image():
    hostaddress=config('ISAPI_HOST')
    print(f'downloading http://{hostaddress}/ISAPI/Streaming/channels/101/picture')
    imageResult = requests.get(f'http://{hostaddress}/ISAPI/Streaming/channels/101/picture', auth=HTTPDigestAuth(config('ISAPI_USER'), config('ISAPI_PASSWORD')))
    print(imageResult.content)

download_image()