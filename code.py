from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import numpy as np
import time

# Video input
cap = cv2.VideoCapture('highway.mp4')

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# COCO class labels
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
              "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
              "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
              "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
              "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
              "scissors", "teddy bear", "hair drier", "toothbrush"]

# Load mask image
mask = cv2.imread('mask.png')

# Initialize SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Red line for counting
limits = [500, 750, 700, 750]

# FPS and pixel/meter conversion
fps = cap.get(cv2.CAP_PROP_FPS)
PIXEL_PER_METER = 10  # adjust for real-world size

# For tracking count and speed
totalCount = []
speed_dict = {}

while True:
    success, img = cap.read()
    if not success:
        break

    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)

    detection = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Detect only specific vehicle classes
            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.5:
                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), scale=0.6, thickness=1)
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detection = np.vstack((detection, currentArray))

    resultsTracker = tracker.update(detection)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f'{int(Id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)

        # Vehicle counting logic
        if limits[0] < cx < limits[2] and abs(cy - limits[1]) < 15:
            if Id not in totalCount:
                totalCount.append(Id)

        # Speed estimation logic
        if Id not in speed_dict:
            speed_dict[Id] = {'prev_cy': cy, 'prev_time': time.time(), 'speed': 0}
        else:
            prev_cy = speed_dict[Id]['prev_cy']
            prev_time = speed_dict[Id]['prev_time']
            curr_time = time.time()
            delta_time = curr_time - prev_time
            delta_pixels = abs(cy - prev_cy)
            delta_meters = delta_pixels / PIXEL_PER_METER
            speed = delta_meters / delta_time  # m/s
            speed_kmph = speed * 3.6
            speed_dict[Id] = {'prev_cy': cy, 'prev_time': curr_time, 'speed': speed_kmph}

            # Show speed on frame
            cvzone.putTextRect(img, f"{int(speed_kmph)} km/h", (x1, y2 + 30), scale=0.8, thickness=2)

    # Show total vehicle count
    cvzone.putTextRect(img, f"Count: {len(totalCount)}", (50, 50), scale=2, thickness=3, colorR=(0, 255, 0))

    # Show video
    scaled_img = cv2.resize(img, (960, 540))
    scaled_region = cv2.resize(imgRegion, (960, 540))
    cv2.imshow('frame', scaled_img)
    cv2.imshow("imageRegion", scaled_region)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
