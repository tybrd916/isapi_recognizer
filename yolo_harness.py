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
        "cameras": [
            {"name": "backyard"},
            {"name": "driveway"}
        ],
        "camera_sequence": ["driveway","backyard"]
    }

    def __init__(self):
        print(self.configDict)
        self.cameraLoop()

    def cameraLoop(self):
        numCameras = len(self.configDict["cameras"]) if "camera_sequence" not in self.configDict else len(self.configDict["camera_sequence"])
        print(numCameras)

yh = yolo_harness()