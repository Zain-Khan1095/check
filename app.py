import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import time
import random
from collections import deque

# -------------------------------
# Cached model loader (use medium for cloud)
# -------------------------------
@st.cache_resource
def get_proctor_model():
    return ProctorModel('m')  # medium – balances accuracy and memory

# -------------------------------
# Enhanced Proctoring Model
# -------------------------------
class ProctorModel:
    def __init__(self, model_size='m'):
        st.info(f"📦 Loading YOLOv8{model_size} model (first time may take a while)...")
        self.yolo = YOLO(f'yolov8{model_size}.pt')
        
        # MediaPipe for head pose
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=2,
                min_detection_confidence=0.5
            )
            self.use_mediapipe = True
        except Exception as e:
            self.use_mediapipe = False
            st.warning(f"MediaPipe not available: {e} – head pose detection disabled.")
        
        # Suspicious objects (COCO classes)
        self.suspicious = {
            67: 'cell phone', 73: 'laptop', 72: 'mouse', 76: 'keyboard',
            62: 'tv/monitor', 77: 'book', 78: 'notebook', 74: 'smartwatch',
        }
        self.conf_thresh = 0.5
        self.head_turn_thresh = 0.04

    def analyze(self, img_bgr):
        """Return list of violations."""
        results = self.yolo(img_bgr, verbose=False, conf=self.conf_thresh)[0]
        
        # Count people
        person_count = 0
        for box in results.boxes:
            if results.names[int(box.cls[0])] == 'person':
                person_count += 1
        
        # Suspicious objects
        objects = set()
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if conf >= self.conf_thresh and cls in self.suspicious:
                objects.add(self.suspicious[cls])
        
        # Head pose
        looking_away = False
        if person_count == 1 and self.use_mediapipe:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            faces = self.face_mesh.process(rgb)
            if faces and faces.multi_face_landmarks:
                lm = faces.multi_face_landmarks[0].landmark
                nose = lm[4]
                left_eye = lm[33]
                right_eye = lm[263]
                eye_center_x = (left_eye.x + right_eye.x) / 2
                if abs(nose.x - eye_center_x) > self.head_turn_thresh:
                    looking_away = True
        
        violations = []
        if person_count > 1:
            violations.append(f"Multiple people ({person_count})")
        if person_count == 0:
            violations.append("No person detected")
        if objects:
            violations.append(f"Suspicious: {', '.join(objects)}")
        if looking_away:
            violations.append("Looking away")
        return violations

# -------------------------------
# Video Processor
# -------------------------------
class ProctorVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = get_proctor_model()
        self.last_capture = time.time()
        self.capture_interval = random.randint(5, 15)
        self.violation_log = deque(maxlen=20)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        now = time.time()
        if now - self.last_capture > self.capture_interval:
            violations = self.model.analyze(img)
            if violations:
                self.violation_log.appendleft({
                    'time': time.strftime("%H:%M:%S"),
                    'msg': " | ".join(violations)
                })
            self.last_capture = now
            self.capture_interval = random.randint(5, 15)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -------------------------------
# Streamlit UI (unchanged from before)
# -------------------------------
st.set_page_config(page_title="AI Proctored Quiz", layout="wide")
st.title("📝 AI-Proctored Quiz")

# ... (rest of your UI code stays exactly as you had it)
