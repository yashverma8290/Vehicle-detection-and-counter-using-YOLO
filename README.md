====================================================================
🚗 VEHICLE DETECTION, TRACKING, COUNTING & SPEED ESTIMATION - YOLOv8
====================================================================

This is a complete, single-file project that performs:

✅ Real-time object detection using YOLOv8  
✅ Vehicle tracking using a simple built-in tracker (no external files)  
✅ Vehicle counting when crossing a virtual line  
✅ Speed estimation (in km/h) based on pixel displacement and FPS  
✅ Display using OpenCV and CvZone  
✅ All code inside ONE Python file — no YAML, no sort.py, no extra configs  

--------------------------------------------------
📁 REQUIREMENTS
--------------------------------------------------

Install these Python packages:

    pip install ultralytics opencv-python cvzone numpy

--------------------------------------------------
📁 FILES IN THIS PROJECT
--------------------------------------------------

1. README.txt         ← You're reading this
2. vehicle_tracking.py ← Main Python script (save your code here)
3. highway.mp4        ← Your video file (or use webcam)

✅ No need for: sort.py, config.yaml, weights download — YOLOv8 will auto-download.

--------------------------------------------------
▶️ HOW TO RUN
--------------------------------------------------

1. Place your video file (e.g., highway.mp4) in the same folder.

2. Save your full Python script as:
       vehicle_tracking.py

3. Run the script:

       python vehicle_tracking.py

4. Press 'q' to quit.

--------------------------------------------------
🧠 WHAT THIS DOES
--------------------------------------------------

- Loads YOLOv8 Nano model (yolov8n.pt)
- Detects objects: car, truck, bus, motorbike
- Filters detections using a polygon ROI
- Assigns a unique ID to each vehicle
- Tracks vehicle across frames
- Counts vehicles crossing a red line
- Estimates and displays vehicle speed in km/h

--------------------------------------------------
⚙️ SPEED ESTIMATION
--------------------------------------------------

Speed is calculated as:

    Speed = (Displacement in pixels / Pixels per meter) / Time

Then converted to km/h using:

    Speed (km/h) = Speed (m/s) × 3.6

You can tune the value in your code:

    PIXELS_PER_METER = 10.0

Adjust this based on your video resolution and real-world scale.

--------------------------------------------------
📊 TYPICAL OUTPUT ON VIDEO FRAME
--------------------------------------------------

- Class label and confidence (e.g., Car 0.91)
- Unique ID per vehicle (e.g., ID: 5)
- Speed (e.g., 67 km/h)
- Vehicle count on screen (e.g., Count: 12)

--------------------------------------------------
✅ FEATURES SUMMARY
--------------------------------------------------

- [✔] YOLOv8 Nano detection (ultralytics)
- [✔] Built-in ID-based vehicle tracking (centroid method)
- [✔] Speed estimation in km/h (frame-based)
- [✔] Vehicle counting with red line crossing
- [✔] ROI filtering using polygon
- [✔] All in one file (no YAML, no sort.py)

--------------------------------------------------
📌 TO-DO / FUTURE IMPROVEMENTS
--------------------------------------------------

- [ ] Add CSV logging (vehicle ID, speed, timestamp)
- [ ] Add speed violation alerts
- [ ] Build Streamlit or Flask dashboard
- [ ] Multi-line and multi-lane support
- [ ] Save vehicle images or video clips



--------------------------------------------------
👋 THANK YOU
--------------------------------------------------

Enjoy real-time detection, counting, and speed analysis with a simple Python file and YOLOv8!
