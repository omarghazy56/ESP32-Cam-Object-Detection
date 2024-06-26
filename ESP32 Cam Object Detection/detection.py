import cv2
import numpy as np
import time

# Set up GPU inference
cv2.ocl.setUseOpenCL(True)
cv2.dnn_registerLayer('Region', cv2.dnn.RegionLayer)

desired_fps = 60
width = 320
height = 320

# Load the model
net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb',
                                    'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')

# Set backend and target to CUDA
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

classNames = []
with open('coco.names', 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

required_id = 5
print("started")
winName = "detection"


def trace(classIds, bbox, required_id):
    for idx, id in enumerate(classIds):
        if id == required_id:
            coordinates = bbox[idx]
            x = abs(coordinates[2] - coordinates[0])
            y = abs(coordinates[3] - coordinates[1])
            cx = width//2
            cy = height//2
            if abs(x-cx) > 10:
                if x < cx:
                    return "right"
                if x > cx:
                    return "left"
            if abs(y-cy) > 10:
                if y < cy:
                    return "up"
                if y > cy:
                    return "down"
    pass


cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

while True:
    start_time = time.time()
    _, frame = cap.read()
    classIds, confs, bbox = net.detect(frame, confThreshold=0.5)
    direction = trace(classIds, bbox, required_id)
    print(direction)
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if classId != required_id:
                continue
            cv2.rectangle(frame, box, color=(0, 255, 0), thickness=3)
            cv2.putText(frame, classNames[classId-1], (box[0]+10, box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow(winName, frame)
    processing_time = time.time() - start_time
    delay = max(1, int((1/desired_fps - processing_time) * 1000))
    tecla = cv2.waitKey(delay) & 0xFF
    if tecla == 27:
        break

cv2.destroyAllWindows()
