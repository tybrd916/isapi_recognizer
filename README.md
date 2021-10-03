### Security Camera Person/Vehicle recognizer ###

Project to analyze an ISAPI "stream" by polling on a specified frequency (i.e every 1, 5, 10 seconds).

Compare each image to prior, and save a copy if significant differences exist:
1) At least one more human recognized in image
2) At least one more animal recognized in image
3) At least one more vehicle recognized in image

### Implementation Details:
- Implementation in Python 3.9 because of YOLO
  - wget https://www.python.org/ftp/python/3.9.6/Python-3.9.6.tgz
  - tar -zxf Python-3.9.6.tgz
  - cd Python-3.9.6; ./configure; make;
- Considering using OpenCV (Open Computer Vision)
- Considering using YOLO (you only look once) recognition/categorization libraries
  - Current - https://github.com/ultralytics/yolov5
    - git clone https://github.com/ultralytics/yolov5
    - cd yolov5
    - python3 -m pip install -r requirements.txt
  - Older Darknet - https://pjreddie.com/darknet/yolo/

### Using ISAPI HTTP endpoints like:
http://<username>:<password>@<ip-address>/ISAPI/Streaming/channels/101/picture