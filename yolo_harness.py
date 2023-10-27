# Scan any HTTP camera source still image
#  Include option for HTTP Auth parameters being passed securely
# Parse photos in camera specific subdirectories
#  Configurable via set-up file with camera URL/credentials, directory name
#  Configure camera specific "blind spots" to ignore
#  Capture photos only when specific classes of objects are detected
#    Ignore if the same object remains in frame in approximately the same position
#    Ignore if the same count of objects has been in the past X many frames (camera specific lookback memory)
#  Email alerts for specific classes of objects

class yolo_harness:
    configDict = {
        "cameras": {
            "backyard": {"blindspots": [],
                         "objects_of_interest": ["person","bicycle","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"],
                         "url": "",
                         "user": "",
                         "password": ""
                        },
            "driveway": {"blindspots": [(0.0,0.2),(1.0,0.2)],
                         "objects_of_interest": ["car","motorcycle","bus","train","truck","person","bicycle","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"],
                         "url": "",
                         "user": "",
                         "password": ""
                        }
        },
        "xxcamera_sequence": ["driveway","backyard"]
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
        self.cameraLoop()

    def cameraLoop(self):
        numCameras = len(self.configDict["camera_sequence"])
        currentCamera = 0
        while True: #infinite loop
            currentCamera=currentCamera+1
            if(currentCamera >= numCameras):
                currentCamera=0
            currentCameraName = self.configDict["camera_sequence"][currentCamera]
            cameraConfig = self.configDict["cameras"][currentCameraName]
            print(cameraConfig)


yh = yolo_harness()