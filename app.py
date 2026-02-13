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
# Enhanced Proctoring Model
# -------------------------------
class ProctorModel:
    def __init__(self, model_size='x'):  # 'x' for extra large (best accuracy)
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
        except:
            self.use_mediapipe = False
            st.warning("MediaPipe not available – head pose detection disabled.")
        
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
# Video Processor for WebRTC
# -------------------------------
class ProctorVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = ProctorModel('m')  # use extra large for best accuracy
        self.last_capture = time.time()
        self.capture_interval = random.randint(5, 15)
        self.violation_log = deque(maxlen=20)  # keep last 20 alerts

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Random capture
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
        
        # Draw bounding boxes (optional – can slow down)
        # For performance, we skip drawing here.
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="AI Proctored Quiz", layout="wide")
st.title("📝 AI-Proctored Quiz")

# Initialize session state for quiz
if 'answers' not in st.session_state:
    st.session_state.answers = [-1] * 3  # 3 questions
if 'current_q' not in st.session_state:
    st.session_state.current_q = 0

# Quiz data
questions = [
    {"q": "What is the capital of France?", 
     "options": ["Berlin", "Madrid", "Paris", "Rome"], 
     "answer": 2},
    {"q": "Which planet is known as the Red Planet?", 
     "options": ["Earth", "Mars", "Jupiter", "Saturn"], 
     "answer": 1},
    {"q": "Who wrote 'Hamlet'?", 
     "options": ["Dickens", "Hemingway", "Shakespeare", "Tolkien"], 
     "answer": 2},
]

# Layout: left column for quiz, right for proctoring
col1, col2 = st.columns(2)

with col1:
    st.header("Quiz")
    q = questions[st.session_state.current_q]
    st.subheader(f"Q{st.session_state.current_q+1}: {q['q']}")
    
    # Options
    opt = st.radio("Select one:", q['options'], 
                   index=st.session_state.answers[st.session_state.current_q] if st.session_state.answers[st.session_state.current_q] != -1 else 0)
    
    # Save answer when changed
    st.session_state.answers[st.session_state.current_q] = q['options'].index(opt)
    
    # Navigation
    col_prev, col_next, _ = st.columns([1,1,2])
    with col_prev:
        if st.button("⬅ Previous") and st.session_state.current_q > 0:
            st.session_state.current_q -= 1
            st.rerun()
    with col_next:
        if st.button("Next ➔") and st.session_state.current_q < len(questions)-1:
            st.session_state.current_q += 1
            st.rerun()
    
    # Submit
    if st.button("Submit Quiz", type="primary"):
        score = 0
        for i, ans in enumerate(st.session_state.answers):
            if ans == questions[i]['answer']:
                score += 1
        st.success(f"Your score: {score}/{len(questions)}")
        # Optionally stop proctoring here – but we'll keep it running

with col2:
    st.header("Live Proctoring")
    st.caption("Random snapshots are analyzed for violations.")
    
    # WebRTC streamer
    ctx = webrtc_streamer(
        key="proctor",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=ProctorVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Display violation log
    if ctx.video_processor:
        log = ctx.video_processor.violation_log
        if log:
            st.subheader("Alerts Log")
            for entry in log:
                st.warning(f"[{entry['time']}] {entry['msg']}")
        else:
            st.info("No violations detected yet.")
    else:

        st.info("Waiting for webcam...")
