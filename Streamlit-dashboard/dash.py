import streamlit as st
import tempfile
import cv2
import os

# Import your AI model functions
from ultralytics import YOLO
import cvzone
import math
import numpy as np

# Function for fire detection
def fire_detection(video_path, display_frame):
    cap = cv2.VideoCapture(video_path)
    model = YOLO("../Fire-trained-model/best (1) (1).pt")
    classNames = ['fire', 'smoke', 'other']
    frame_skip = 5
    frame_count = 0

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break
        frame_count += 1
        if frame_count % frame_skip == 0:
            results = model(img, stream=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(img, (x1, y1, w, h))
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(30, y1)))
            display_frame.image(img, channels="BGR")
    cap.release()

# Function for PPE detection
def ppe_detection(video_path, display_frame):
    cap = cv2.VideoCapture(video_path)
    model = YOLO("../PPE-trained-model/best (1).pt")
    classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                cvzone.putTextRect(img, f'{classNames[cls]}{conf}', (max(0, x1), max(30, y1)))
        display_frame.image(img, channels="BGR")
    cap.release()

# Function for camera blockage detection
def camera_blockage(video_path, display_frame):
    BRIGHTNESS_THRESHOLD = 20
    def get_average_brightness(frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray_frame)
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        avg_brightness = get_average_brightness(frame)
        if avg_brightness < BRIGHTNESS_THRESHOLD:
            st.error("Alert: Camera blocked!")
        display_frame.image(frame, channels="BGR")
    cap.release()

# Function for fall detection
def fall_detection(video_path, display_frame):
    cap = cv2.VideoCapture(video_path)
    model = YOLO('../Fall-Detection/best (2) (1).pt')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(frame, (x1, y1, w, h))
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                label = f'{model.names[cls]} {conf:.2f}'
                cvzone.putTextRect(frame, label, (max(0, x1), max(30, y1)))
                if model.names[cls] == 'person':
                    if h < w:
                        cv2.putText(frame, "Fall Detected", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        display_frame.image(frame, channels="BGR")
    cap.release()

# Function for anomaly detection
def anomaly_detection(video_path, display_frame):
    cap = cv2.VideoCapture(video_path)
    prev_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if prev_frame is not None:
            gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff_frame = cv2.absdiff(gray_prev, gray_curr)
            _, thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 500:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, "Anomaly Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        prev_frame = frame.copy()
        display_frame.image(frame, channels="BGR")
    cap.release()

# Function for camera rotation alert
def camera_rotation_alert(video_path, display_frame):
    cap = cv2.VideoCapture(video_path)
    prev_mean_angle = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        mean_angle = np.mean(angle)
        if abs(mean_angle - prev_mean_angle) > 15:
            cv2.putText(frame, "Camera Rotated", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        prev_mean_angle = mean_angle
        display_frame.image(frame, channels="BGR")
    cap.release()

# Streamlit UI
st.title("AI Video Processing Dashboard")
st.write("Provide a video file or RTSP stream URL and select the AI models to apply.")

video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
video_url = st.text_input("Or provide a RTSP stream URL")

apply_fire_detection = st.checkbox("Apply Fire Detection")
apply_ppe_detection = st.checkbox("Apply PPE Detection")
apply_camera_blockage_detection = st.checkbox("Apply Camera Blockage Detection")
apply_fall_detection = st.checkbox("Apply Fall Detection")
apply_anomaly_detection = st.checkbox("Apply Anomaly Detection")
apply_camera_rotation_alert = st.checkbox("Apply Camera Rotation Alert")

if st.button("Submit"):
    if video_file:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(video_file.read())
        video_path = temp_file.name
    elif video_url:
        video_path = video_url
    else:
        st.error("Please provide a video file or RTSP stream URL.")
        st.stop()

    st.write("Processing...")

    display_frame = st.empty()

    if apply_fire_detection:
        st.write("Applying Fire Detection...")
        fire_detection(video_path, display_frame)

    if apply_ppe_detection:
        st.write("Applying PPE Detection...")
        ppe_detection(video_path, display_frame)

    if apply_camera_blockage_detection:
        st.write("Applying Camera Blockage Detection...")
        camera_blockage(video_path, display_frame)

    if apply_fall_detection:
        st.write("Applying Fall Detection...")
        fall_detection(video_path, display_frame)

    if apply_anomaly_detection:
        st.write("Applying Anomaly Detection...")
        anomaly_detection(video_path, display_frame)

    if apply_camera_rotation_alert:
        st.write("Applying Camera Rotation Alert...")
        camera_rotation_alert(video_path, display_frame)

    st.success("Processing completed.")
