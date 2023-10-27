# Scan any HTTP camera source still image
#  Include option for HTTP Auth parameters being passed securely
# Parse photos in camera specific subdirectories
#  Configurable via set-up file with camera URL/credentials, directory name
#  Configure camera specific "blind spots" to ignore
#  Capture photos only when specific classes of objects are detected
#    Ignore if the same object remains in frame in approximately the same position
#    Ignore if the same count of objects has been in the past X many frames (camera specific lookback memory)
#  Email alerts for specific classes of objects

import requests
from requests.auth import HTTPDigestAuth
import tempfile
from PIL import Image
from decouple import config
import io

class yolo_harness:
    configDict = {
        "cameras": {
            # "backyard": {"blindspots": [],
            #              "objects_of_interest": ["person","bicycle","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"],
            #              "url": "http://192.168.254.11/ISAPI/Streaming/Channels/101/picture",
            #              "user": config('CAM_USER'),
            #              "password": config('CAM_PASSWORD')
            #             },
            "sideyard": {"blindspots": [],
                         "objects_of_interest": ["person","bicycle","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"],
                         "url": "http://192.168.254.5/ISAPI/Streaming/Channels/101/picture",
                         "user": config('CAM_USER'),
                         "password": config('CAM_PASSWORD')
                        },
            # "driveway": {"blindspots": [(0.0,0.2),(1.0,0.2)],
            #              "objects_of_interest": ["car","motorcycle","bus","train","truck","person","bicycle","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"],
            #              "url": "http://192.168.254.2/ISAPI/Streaming/Channels/101/picture",
            #              "user": config('CAM_USER'),
            #              "password": config('CAM_PASSWORD')
            #             }
        },
        # "camera_sequence": ["driveway","backyard"]
    }
    lastFrameDict = {
        #Store information about the objects in most recent frame (and lookback period too) for each camera name "key": (and persist/reload from disk for when program restarts)
    }

    def __init__(self):
        # Initialize camera_sequence if not already configured
        if "camera_sequence" not in self.configDict:
            camera_sequence = []
            for key in self.configDict["cameras"]:
                camera_sequence.append(key)
            self.configDict["camera_sequence"] = camera_sequence
        print(self.configDict)
        self.cameraLoop()

    def cameraLoop(self):
        numCameras = len(self.configDict["camera_sequence"])
        currentCamera = 0
        # while True: #infinite loop
        currentCamera=currentCamera+1
        if(currentCamera >= numCameras):
            currentCamera=0
        currentCameraName = self.configDict["camera_sequence"][currentCamera]
        cameraConfig = self.configDict["cameras"][currentCameraName]
        self.processCameraImage(currentCameraName, cameraConfig)

    def processCameraImage(self, currentCameraName, cameraConfig):
        image = self.download_image(cameraConfig["url"], cameraConfig["user"], cameraConfig["password"])
        print(image)

    def download_image(self, url, username, password):
        buffer = tempfile.SpooledTemporaryFile(max_size=1e9)
        i = None
        r = None
        if username == "" or password == "":
            r = requests.get(url, stream=True)    
        else:
            r = requests.get(url, auth=HTTPDigestAuth(username, password), stream=True)
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

yh = yolo_harness()