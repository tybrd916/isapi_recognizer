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
                         "objects_of_interest": ["person","bicycle","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"]
                        },
            "driveway": {"blindspots": [(0.0,0.2),(1.0,0.2)],
                         "objects_of_interest": ["car","motorcycle","bus","train","truck","person","bicycle","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"]
                        }
        },
        "camera_sequence": ["driveway","backyard"]
    }
    lastFrameDict = {
        #Store information about the last frame for each camera name "key": (and persist/reload from disk for when program restarts)
    }

    def __init__(self):
        print(self.configDict)
        self.cameraLoop()

    def cameraLoop(self):
        numCameras = len(self.configDict["cameras"]) if "camera_sequence" not in self.configDict else len(self.configDict["camera_sequence"])
        print(numCameras)

yh = yolo_harness()