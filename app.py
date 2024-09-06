import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLOv10

# URL of the YOLO dataset
#url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt"
YOLO_MODEL = "yolov10n.pt"

# Load YOLOv10 model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5x')  # Placeholder for YOLOv10
#model = YOLO.from_pretrained(YOLO_MODEL)
model = YOLOv10(YOLO_MODEL)

# Streamlit app layout
st.set_page_config(layout="wide")

# Sidebar for user inputs
st.sidebar.header("Object Detection Settings")
object_list = st.sidebar.text_input("Enter objects to identify (comma-separated)", "person,car")
tracked_objects = [obj.strip() for obj in object_list.split(",")]

# State to track currently selected stream
if 'selected_stream' not in st.session_state:
    st.session_state['selected_stream'] = None

# Video streams (mocked for example)
video_sources = ['video1.mp4', 'video2.mp4', 'video3.mp4', 'video4.mp4',
                 'video5.mp4', 'video6.mp4', 'video7.mp4', 'video8.mp4']

# Function to process frames with YOLO
def detect_objects(frame):
    results = model(frame)
    detected_objects = results.pandas().xyxy[0][results.pandas().xyxy[0]['name'].isin(tracked_objects)]
    return detected_objects

# Function to display and handle video streams
def display_streams():
    columns = st.columns(4)
    for i, video_source in enumerate(video_sources[:8]):
        if i % 4 == 0 and i != 0:
            columns = st.columns(4)
        with columns[i % 4]:
            cap = cv2.VideoCapture(video_source)
            ret, frame = cap.read()
            if not ret:
                st.warning(f"Stream {video_source} not available")
                continue
            
            detections = detect_objects(frame)
            for _, row in detections.iterrows():
                cv2.rectangle(frame, (int(row['xmin']), int(row['ymin'])), (int(row['xmax']), int(row['ymax'])), (255, 0, 0), 2)
                cv2.putText(frame, f"{row['name']} {row['confidence']:.2f}", (int(row['xmin']), int(row['ymin']) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Convert image to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption=f"Stream {i+1}", use_column_width=True)

            # If clicked, enlarge stream
            if st.button(f"Enlarge Stream {i+1}", key=f"enlarge_{i}"):
                st.session_state['selected_stream'] = video_source

# Function to handle enlarged stream
def display_enlarged_stream(video_source):
    st.header(f"Stream: {video_source}")
    cap = cv2.VideoCapture(video_source)
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning(f"Stream {video_source} not available")
            break

        detections = detect_objects(frame)
        for _, row in detections.iterrows():
            cv2.rectangle(frame, (int(row['xmin']), int(row['ymin'])), (int(row['xmax']), int(row['ymax'])), (255, 0, 0), 2)
            cv2.putText(frame, f"{row['name']} {row['confidence']:.2f}", (int(row['xmin']), int(row['ymin']) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption=f"Stream {video_source}", use_column_width=True)
        
        if st.button("Back to All Streams"):
            st.session_state['selected_stream'] = None
            break

# Main display logic
if st.session_state['selected_stream'] is None:
    display_streams()
else:
    display_enlarged_stream(st.session_state['selected_stream'])
