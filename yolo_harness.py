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
import sys
sys.path.insert(0, './yolov5')
import torch
import time
import json

class yolo_harness:
    configDict = {
        "cameras": {
            # "sideyard": {"blindspots": [],
            #              "objects_of_interest": ["person","bicycle","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"],
            #              "url": "http://192.168.254.5/ISAPI/Streaming/Channels/101/picture",
            #              "user": config('CAM_USER'),
            #              "password": config('CAM_PASSWORD'),
            #              "maxSnapshotsToKeep": 150,
            #             },
            "randomtraffic": {"blindspots": [((0.0,0.5),(1.0,1.0))],
                         "objects_of_interest": ["traffic light", "car", "person","bicycle","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"],
                        #  "url": "https://coe.everettwa.gov/Broadway/Images/Pacific_Oakes/Pacific_Oakes.jpg",
                         "url": "https://coe.everettwa.gov/Broadway/Images/Broadway_Hewitt/Broadway_Hewitt.jpg",
                         "user": "",
                         "password": "",
                         "maxSnapshotsToKeep": 150,
                        },
            # "backyard": {"blindspots": [],
            #              "objects_of_interest": ["person","bicycle","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"],
            #              "url": "http://192.168.254.11/ISAPI/Streaming/Channels/101/picture",
            #              "user": config('CAM_USER'),
            #              "password": config('CAM_PASSWORD'),
            #              "maxSnapshotsToKeep": 150,
            #             },
            # "driveway": {"blindspots": [((0.0,0.2),(1.0,0.2))],
            #              "objects_of_interest": ["car","motorcycle","bus","train","truck","person","bicycle","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"],
            #              "url": "http://192.168.254.2/ISAPI/Streaming/Channels/101/picture",
            #              "user": config('CAM_USER'),
            #              "password": config('CAM_PASSWORD'),
            #              "maxSnapshotsToKeep": 150,
            #             }
        },
        "minConfidence": 0.65,
        "objectBoundaryFuzzyMatch": 0.1,
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
        # initialize yolo model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m')  # or yolov5m, yolov5l, yolov5x, custom

        self.model.conf = float(self.configDict["minConfidence"])
        self.maximumSnapshots = 150
        self.cameraLoop()

    def cameraLoop(self):
        print("Starting Camera watching loop")
        numCameras = len(self.configDict["camera_sequence"])
        currentCamera = 0
        while True: #infinite loop
            currentCamera=currentCamera+1
            if(currentCamera >= numCameras):
                currentCamera=0
            currentCameraName = self.configDict["camera_sequence"][currentCamera]
            cameraConfig = self.configDict["cameras"][currentCameraName]
            self.processCameraImage(currentCameraName, cameraConfig)
            time.sleep(2)

    def processCameraImage(self, currentCameraName, cameraConfig):
        image = self.download_image(cameraConfig["url"], cameraConfig["user"], cameraConfig["password"])
        if image == None:
            print(f"failed to download image from {cameraConfig['+url']}")
            return #avoid crash when no image is returned
        results = self.model(image)
        print(results)
        if currentCameraName not in self.lastFrameDict:
            self.lastFrameDict[currentCameraName] = {}
            self.lastFrameDict[currentCameraName]["lastObjectsDetected"] = None
        # Loop over yolo results
        objectsDetected = {}
        for i, (im, pred) in enumerate(zip(results.ims, results.pred)):
            for xLeft, yTop, xRight, yBottom, conf, cls in reversed(pred):
                detectedName=results.names[int(cls)]
                if detectedName in cameraConfig["objects_of_interest"]:
                    if detectedName not in objectsDetected:
                        objectsDetected[detectedName]=[]
                    objectDetected = {}
                    objectDetected["topLeft"] = (int(xLeft),int(yTop))
                    objectDetected["bottomRight"] = (int(xRight),int(yBottom))
                    objectDetected["topLeftPercent"] = (int(xLeft)/image.width,int(yTop)/image.height)
                    objectDetected["bottomRightPercent"] = (int(xRight)/image.width,int(yBottom)/image.height)
                    objectDetected["confidence"] = float(conf)
                    #Perform blindspot region detection
                    for blindspot in cameraConfig["blindspots"]:
                        if blindspot[0][0] < objectDetected["topLeftPercent"][0] and blindspot[0][1] < objectDetected["topLeftPercent"][1] and blindspot[1][0] > objectDetected["bottomRightPercent"][0] and blindspot[1][1] > objectDetected["bottomRightPercent"][1]:
                            print(f"{detectedName} is in blindspot {blindspot}")
                            objectDetected["withinBlindSpot"]=True
                    objectsDetected[detectedName].append(objectDetected)

        for key in objectsDetected:
            for i, obj in enumerate(objectsDetected[key]):
                #Compare objects detected to most recent frame from camera
                if self.lastFrameDict[currentCameraName]["lastObjectsDetected"]:
                    previousObjects = self.lastFrameDict[currentCameraName]["lastObjectsDetected"]
                    if key in previousObjects:
                        topLeftX_max = float(obj["topLeftPercent"][0])+float(self.configDict["objectBoundaryFuzzyMatch"])
                        topLeftX_min = float(obj["topLeftPercent"][0])-float(self.configDict["objectBoundaryFuzzyMatch"])
                        topLeftY_max = float(obj["topLeftPercent"][1])+float(self.configDict["objectBoundaryFuzzyMatch"])
                        topLeftY_min = float(obj["topLeftPercent"][1])-float(self.configDict["objectBoundaryFuzzyMatch"])
                        bottomRightX_max = float(obj["bottomRightPercent"][0])+float(self.configDict["objectBoundaryFuzzyMatch"])
                        bottomRightX_min = float(obj["bottomRightPercent"][0])-float(self.configDict["objectBoundaryFuzzyMatch"])
                        bottomRightY_max = float(obj["bottomRightPercent"][1])+float(self.configDict["objectBoundaryFuzzyMatch"])
                        bottomRightY_min = float(obj["bottomRightPercent"][1])-float(self.configDict["objectBoundaryFuzzyMatch"])
                        for prevobj in previousObjects[key]:
                            if prevobj["topLeftPercent"][0] > topLeftX_min and prevobj["topLeftPercent"][0] < topLeftX_max and \
                                prevobj["topLeftPercent"][1] > topLeftY_min and prevobj["topLeftPercent"][1] < topLeftY_max and \
                                prevobj["bottomRightPercent"][0] > bottomRightX_min and prevobj["bottomRightPercent"][0] < bottomRightX_max and \
                                prevobj["bottomRightPercent"][1] > bottomRightY_min and prevobj["bottomRightPercent"][1] < bottomRightY_max:
                                # print(f"{key} already seen!")
                                objectsDetected[key][i]["dejavu"]=True
        self.lastFrameDict[currentCameraName]["lastObjectsDetected"]=objectsDetected
        print(json.dumps(objectsDetected, indent=1))

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