# Scan any HTTP camera source still image
#  TODO: Add getting still images from RTSP streams since one of Ted's camera's doesn't give HTTP still image
#  - (https://github.com/Cacsjep/pyrtsputils/blob/main/snapshot_generator.py)
#  - rtsp://username:password@8.0.0.41:554/Streaming/Channels/102
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
from PIL import Image, ImageDraw, ImageFont
from decouple import config
import io
import av
import glob
import os
import sys
sys.path.insert(0, './yolov5')
import torch
import time
import datetime
import json
import re
# from yolov5.utils.plots import Annotator, colors

def timer_func(func): 
    # This function shows the execution time of  
    # the function object passed 
    def wrap_func(*args, **kwargs): 
        t1 = time.process_time()
        result = func(*args, **kwargs) 
        t2 = time.process_time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s') 
        return result 
    return wrap_func  
class yolo_harness:
    configDict = {
        "cameras": {
            # "sideyard": {"cameraGroups": {"driveway":{"blindspots": []}},
            #              "objects_of_interest": ["person","bicycle","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"],
            #              "url": "http://192.168.254.5/ISAPI/Streaming/Channels/101/picture",
            #              "user": config('CAM_USER'),
            #              "password": config('CAM_PASSWORD'),
            #              "maxSnapshotsToKeep": 150,
            #             },
            # "randomtraffic": {"cameraGroups": {"everett1":{"blindspots": [((0.0,0.0),(1.0,0.4))]},"everett2":{"blindspots": [((0.0,0.0),(0.2,0.4))]}},
            #              "objects_of_interest": ["traffic light", "car", "person","bicycle","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"],
            #             #  "url": "https://coe.everettwa.gov/Broadway/Images/Pacific_Oakes/Pacific_Oakes.jpg",
            #              "url": "https://coe.everettwa.gov/Broadway/Images/Broadway_Hewitt/Broadway_Hewitt.jpg",
            #              "user": "",
            #              "password": "",
            #              "maxSnapshotsToKeep": 150,
            #             },
            # "backyard": {"cameraGroups": {"driveway":{"blindspots": []}},
            #              "objects_of_interest": ["person","bicycle","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"],
            #              "url": "http://192.168.254.11/ISAPI/Streaming/Channels/101/picture",
            #              "user": config('CAM_USER'),
            #              "password": config('CAM_PASSWORD'),
            #              "maxSnapshotsToKeep": 150,
            #             },
            "driveway": {"cameraGroups": {"driveway":{"blindspots": [((0.0,0.2),(1.0,0.2))]}},
                        #  "objects_of_interest": ["fire hydrant","bench","car","motorcycle","bus","train","truck","person","bicycle","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"],
                         "objects_of_interest": ["car","motorcycle","bus","train","truck","person","bicycle","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"],
                        #  "url": "http://192.168.254.2/ISAPI/Streaming/Channels/101/picture",
                         "url": "rtsp://8.0.0.41:554/Streaming/Channels/102",
                         "user": config('CAM_USER'),
                         "password": config('CAM_PASSWORD'),
                         "maxSnapshotsToKeep": 150,
                        }
        },
        # "minConfidence": 0.65,
        "minConfidence": 0.15,
        "objectBoundaryFuzzyMatch": 0.05,
        "lookbackDepth": 5,
        "maximumSnapshots": 200,
        "maximumSnapshots": 4,
        "saveDirectoryPath": "/tmp/yolo_cams/"
        # "camera_sequence": ["driveway","backyard"]
    }
    lastFrameDict = {
        #Store information about the objects in most recent frame (and lookback period too) for each camera name "key": (and persist/reload from disk for when program restarts)
    }
    interestsDict = {
        #Store information lookback queue for camera/object-class
    }
    videoStreamsDict = {

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
            objectsDetected, image = self.detectImageObjects(currentCameraName, cameraConfig)
            if objectsDetected != None:
                for cameraGroup in cameraConfig["cameraGroups"]:
                    # print(cameraGroup)
                    self.filterNotifications(cameraGroup, currentCameraName, image, objectsDetected)

            # time.sleep(1)

    def filterNotifications(self, cameraGroup, currentCameraName, image, objectsDetected):

        if cameraGroup not in self.interestsDict:
            self.interestsDict[cameraGroup] = {}

        if currentCameraName not in self.interestsDict[cameraGroup]:
            self.interestsDict[cameraGroup][currentCameraName] = {}

        interestsFlagged = []
        for key in objectsDetected:
            if key not in self.interestsDict[cameraGroup][currentCameraName]:
                self.interestsDict[cameraGroup][currentCameraName][key] = {}
                self.interestsDict[cameraGroup][currentCameraName][key]["lookbackQueue"] = [0]
        
            maxObjectCount=max(self.interestsDict[cameraGroup][currentCameraName][key]["lookbackQueue"])
            # visibleObjectList=[1 for x in objectsDetected[key] if "withinBlindSpot" not in x[cameraGroup]] #Do not count objects within cameraGroup's blindspot
            visibleObjectList=[1 for x in objectsDetected[key] if "withinBlindSpot" not in x[cameraGroup] and "dejavu" not in x[cameraGroup]] #Do not count objects within cameraGroup's blindspot or seen in prior frame
            if len(visibleObjectList) > maxObjectCount:
                interestsFlagged.append(f"{len(visibleObjectList)} {key}{'s' if len(visibleObjectList) > 1 else ''}")
            
            #Update Lookback count list
            lookbackDepth = 30
            if "lookbackDepth" in self.configDict:
                lookbackDepth=self.configDict["lookbackDepth"]
            if "lookbackDepth" in self.configDict["cameras"][currentCameraName]:
                lookbackDepth=self.configDict["cameras"][currentCameraName]["lookbackDepth"]
            if len(self.interestsDict[cameraGroup][currentCameraName][key]["lookbackQueue"]) >= lookbackDepth:
                self.interestsDict[cameraGroup][currentCameraName][key]["lookbackQueue"].pop(0)
            self.interestsDict[cameraGroup][currentCameraName][key]["lookbackQueue"].append(len(objectsDetected[key]))

        # Save lastObjectsDetected to state TODO: Merge data at the object level!
        self.lastFrameDict[currentCameraName]["lastObjectsDetected"]=objectsDetected

        if len(interestsFlagged) > 0:
            print(f"{cameraGroup} - {currentCameraName} - {interestsFlagged}")
            self.saveAndNotify(cameraGroup, currentCameraName, image, objectsDetected, interestsFlagged)
            
    @timer_func
    def saveAndNotify(self, cameraGroup, currentCameraName, image, objectsDetected, interestsFlagged):
        txt = Image.new('RGBA', image.size, (255,255,255,0))
        drawtxt = ImageDraw.Draw(txt)

        #TODO: Scale font size to image resolution
        scaleFactor = image.size[0]/720
        font  = ImageFont.truetype("Arial.ttf", int(10*scaleFactor), encoding="unic")
        for objectType in objectsDetected:
            for object in objectsDetected[objectType]:
                if "withinBlindSpot" in object[cameraGroup]:
                    #Note multiply Blindspot percentage tuples by image size tuples
                    drawtxt.rectangle(xy=[tuple([image.size[0]*object[cameraGroup]["withinBlindSpot"][0][0],image.size[1]*object[cameraGroup]["withinBlindSpot"][0][1]]),tuple([image.size[0]*object[cameraGroup]["withinBlindSpot"][1][0],image.size[1]*object[cameraGroup]["withinBlindSpot"][1][1]])], outline=(0,0,0,100), width=int(10*scaleFactor))
                    drawtxt.text( tuple([image.size[0]*object[cameraGroup]["withinBlindSpot"][0][0],image.size[1]*object[cameraGroup]["withinBlindSpot"][0][1]]), f"blindspot", fill=(255,255,255,180), font=font)
                else:
                    textStr=f"{'*' if 'dejavu' in object[cameraGroup] else ''}{objectType} {round(object['confidence'],2)}"
                    textBoxSize=bbox=drawtxt.textbbox( (0,0), textStr, font=font)
                    bbox=None
                    textCoordinate=(0,0)
                    if object["topLeft"][1] < 0.25*image.size[1]:
                        #Put the label on the bottom
                        textCoordinate=(object["topLeft"][0],object["bottomRight"][1]-textBoxSize[1])
                    else:
                        #Put the label on the top
                        textCoordinate=(object["topLeft"][0],object["topLeft"][1]-textBoxSize[3])
                    bbox=drawtxt.textbbox( textCoordinate, textStr, font=font)
                    drawtxt.rectangle(bbox, fill=(0,0,255,100))
                    drawtxt.rectangle(xy=[object["topLeft"],object["bottomRight"]], outline=(0,0,255,100), width=int(1*scaleFactor))
                    drawtxt.text( textCoordinate, textStr, fill=(255,255,255,180), font=font)

        #Save image to filesystem
        dirName=f"{self.configDict['saveDirectoryPath']}/{cameraGroup}/"
        fileName=f"{datetime.datetime.now()} {currentCameraName} {interestsFlagged}"
        # image.save(f"{dirName}/{fileName} original.png","PNG", compress_level=1)
        # txt.save(f"{dirName}/{fileName} overlay.png","PNG", compress_level=1)
        self.clearOldestSnapshots(dirName)
        combined = Image.alpha_composite(image, txt)   
        combined.save(f"{dirName}/{fileName}.png","PNG", compress_level=1)

    def clearOldestSnapshots(self,dirname="snapshots"):
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        snapshotList = glob.glob(dirname+"/*.png")
        numSnapShotsToDelete = len(snapshotList) - int(self.configDict["maximumSnapshots"])
        for snapshotPath in sorted(snapshotList):
            if numSnapShotsToDelete > 0:
                os.remove(snapshotPath)
                numSnapShotsToDelete=numSnapShotsToDelete-1
            else:
                break

    def detectImageObjects(self, currentCameraName, cameraConfig):
        image = self.download_image(currentCameraName, cameraConfig["url"], cameraConfig["user"], cameraConfig["password"])
        if image == None:
            print(f"failed to download image from {cameraConfig['url']}")
            return #avoid crash when no image is returned
        results = self.model(image)
        # print(results)
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
                    for cameraGroup in cameraConfig["cameraGroups"]:
                        objectDetected[cameraGroup] = {}
                        for blindspot in cameraConfig["cameraGroups"][cameraGroup]["blindspots"]:
                            if blindspot[0][0] < objectDetected["topLeftPercent"][0] and blindspot[0][1] < objectDetected["topLeftPercent"][1] and blindspot[1][0] > objectDetected["bottomRightPercent"][0] and blindspot[1][1] > objectDetected["bottomRightPercent"][1]:
                                objectDetected[cameraGroup]["withinBlindSpot"]=blindspot
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
                                objectsDetected[key][i][cameraGroup]["dejavu"]=True
        return objectsDetected, image

    @timer_func
    def download_image(self, cameraName, url, username, password):
        if re.match("^rtsp://", url):
            return self.download_rtsp_image(cameraName, url, username, password)
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
            i = Image.open(io.BytesIO(buffer.read())).convert("RGBA")
            # i.save(os.path.join(out_dir, 'image.jpg'), quality=85)
        else:
            print(r)
        buffer.close()
        return i
    
    def download_rtsp_image(self, cameraName, url, username, password):
        if username != "" and password != "":
            url = re.sub("^rtsp://",f"rtsp://{username}:{password}@", url)
        # print(f' Try to connect to {url}')
        # Connect to RTSP URL
        if cameraName in self.videoStreamsDict:
            video = self.videoStreamsDict[cameraName]
        else:
            video = av.open(url, 'r')
            self.videoStreamsDict[cameraName] = video
        # Iter over Package to get an frame
        i = None
        for packet in video.demux():
            # When frame is decoded
            for frame in packet.decode():
                # Save Frame into JPEG
                if hasattr(frame, 'to_image') and callable(frame.to_image):
                    i = frame.to_image().convert("RGBA")
                    i.save(f"{self.configDict['saveDirectoryPath']}/tyler.png")
                    # Return because we just need one frame
                    return i
        print(f"download_rtsp_image did not find an image for {url}")

yh = yolo_harness()