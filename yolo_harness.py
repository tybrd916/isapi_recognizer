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
import cv2
import glob
import os
import sys
sys.path.insert(0, './yolov5')
import torch
import time
import datetime
import json
import re
import queue, threading
# from yolov5.utils.plots import Annotator, colors
import smtplib
from email.message import EmailMessage
from email.utils import make_msgid
import urllib.request

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

# bufferless VideoCapture
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.lock = threading.Lock()
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    # grab frames as soon as they are available
    def _reader(self):
        while True:
            with self.lock:
                ret = self.cap.grab()
            if not ret:
                break

    # retrieve latest frame
    def read(self):
        with self.lock:
            _, frame = self.cap.retrieve()
        return frame

class yolo_harness:
    configDict = {
        "cameras": {
            "randomtraffic": {"cameraGroups": {"everett1":{"blindspots": [((0.0,0.0),(1.0,0.4))]},"everett2":{"blindspots": [((0.0,0.0),(0.2,0.4))]}},
                         "objects_of_interest": ["traffic light", "car", "person","bicycle","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"],
                         "url": "https://coe.everettwa.gov/Broadway/Images/Broadway_Hewitt/Broadway_Hewitt.jpg",
                         "user": "",
                         "password": "",
                         "maxSnapshotsToKeep": 150,
                        },
        },
        # "minConfidence": 0.65,
        "minConfidence": 0.15,
        "objectBoundaryFuzzyMatch": 0.05,
        "lookbackDepth": 5,
        "maximumSnapshots": 200,
        "maximumSnapshots": 4,
        "saveDirectoryPath": "/tmp/yolo_cams/",
        "camera_sequence": ["randomtraffic"],
        "notifyUrl": "http://192.168.40.250:88/JSON?request=runevent&group=Driveway&name=Announce%20Driveway%20Motion"
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
        if os.path.exists('yolo_config.json'):
            with open('yolo_config.json', 'r') as openfile: 
                # Reading from json file
                self.configDict = json.load(openfile)
        else:
            #Default initial configuration file if missing
            with open("yolo_config.json", "w") as outfile:
                outfile.write(json.dumps(self.configDict, indent=1))

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

            time.sleep(1)

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

        # Scale font size to image resolution
        scaleFactor = image.size[0]/720
        font  = ImageFont.truetype("Arial.ttf", int(20*scaleFactor), encoding="unic")
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
        filePath=f"{dirName}/{fileName}.png"
        combined.save(filePath,"PNG", compress_level=1)
        self.emailImage(filePath, currentCameraName, interestsFlagged, interestsFlagged)

        # tell Homeseer to announce motion
        if('notifyUrl' in self.configDict["cameras"][currentCameraName]):
            try:
                with urllib.request.urlopen(self.configDict["cameras"][currentCameraName]['notifyUrl']) as response:
                    html = response.read()
            except:
                print(f'Error notifying {self.configDict["cameras"][currentCameraName]["notifyUrl"]}')

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
            return None, None #avoid crash when no image is returned
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
        # buffer = tempfile.SpooledTemporaryFile(max_size=1e9)
        i = None
        r = None
        if username == "" or password == "":
            r = requests.get(url, stream=True)    
        else:
            r = self.requests_get(url, auth=HTTPDigestAuth(username, password), stream=True)
        if r.status_code == 200:
            downloaded = 0
            i = Image.open(io.BytesIO(r.raw.read())).convert("RGBA")
            # DEBUG by saving unlabeled image to disk
            # i.save(f"{self.configDict['saveDirectoryPath']}/{cameraName}.png", compress_level=1)
        else:
            print(r)
        # buffer.close()
        return i
    
    @timer_func
    def requests_get(self, url, auth, stream):
        return requests.get(url, auth=auth, stream=stream)

    # Idea from https://stackoverflow.com/a/54755738 to always read the latest from from RTSP stream

    def download_rtsp_image(self, cameraName, url, username, password):
        if username != "" and password != "":
            url = re.sub("^rtsp://",f"rtsp://{username}:{password}@", url)
        # print(f' Try to connect to {url}')
        # Connect to RTSP URL
        if cameraName in self.videoStreamsDict:
            video = self.videoStreamsDict[cameraName]
        else:
            video = VideoCapture(url)
            self.videoStreamsDict[cameraName] = video
            print(f"starting video stream {cameraName}")
            time.sleep(2)
        frame = video.read()
        try:
            color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        except Exception as error:
            print(f"{cameraName} download_rtsp_image failed", error)
            return
        pil_image = Image.fromarray(color_converted).convert("RGBA")
        return pil_image

    # Configure Gmail to allow sending from this app:
    # https://towardsdatascience.com/automate-sending-emails-with-gmail-in-python-449cc0c3c317
    def emailImage(self, filePath, currentCameraName, interestsFlagged, bodyStr):
        if("emailTo" not in self.configDict or len(self.configDict['emailTo']) < 5):
            return; #Do not send e-mail if no EMAIL_TO address is configured
        # attachment = 'latest_thumb.jpg'

        msg = EmailMessage()
        msg["To"] = self.configDict['emailTo'].split(",")
        msg["From"] = self.configDict['emailFrom']
        msg["Subject"] = f"{currentCameraName} Yolo Snapshot - {interestsFlagged}"

        attachment_cid = make_msgid()

        msg.set_content('<b>%s</b><br/><img src="cid:%s"/><br/>' % (bodyStr, attachment_cid), 'html')

        # msg.add_related(imagebytes, 'image', 'jpeg', cid=attachment_cid)
        with open(filePath, 'rb') as fp:
            msg.add_related(
                fp.read(), 'image', 'png', cid=attachment_cid)
       
        # creates SMTP session
        s = smtplib.SMTP('smtp.gmail.com', 587)
       
        # start TLS for security
        s.starttls()
       
        # Authentication
        s.login(self.configDict['emailFrom'], self.configDict['emailPassword'])
       
        # Converts the Multipart msg into a string
        text = msg.as_string()
       
        # sending the mail
        s.sendmail(self.configDict['emailFrom'], self.configDict['emailTo'].split(","), text)

yh = yolo_harness()