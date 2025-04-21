import streamlit as st
import cv2
import numpy as np
import time

# Streamlit page config
st.set_page_config(page_title="Live Face Blur", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(45deg, #6B4226, #D4A276, #A8DADC) !important;
    }
    h1 {
        font-family: 'Verdana', sans-serif !important;
        font-size: 2.5rem !important;
        color: #464646 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        margin: 2rem 0 !important;
        text-align: center;
    }
    .stButton > button {
        padding: 1rem 2.5rem !important;
        font-size: 1.2rem !important;
        font-weight: bold !important;
        border-radius: 20px !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown("<h1>Live Face Blur Using OpenCV</h1>", unsafe_allow_html=True)

# Button state control
if "run" not in st.session_state:
    st.session_state.run = False

# Centered buttons layout
col1, col2, col3 = st.columns([2, 4, 4])
with col1:
    st.write("")  # Spacer
with col2:
    if st.button("🔴 Start Webcam"):
        st.session_state.run = True
with col3:
    if st.button("⛔ Stop Webcam"):
        st.session_state.run = False

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# Frame placeholder
frame_placeholder = st.empty()

# Run webcam loop
if st.session_state.run:
    cap = cv2.VideoCapture(0)
    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Unable to access the camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = []

        # Detect frontal faces
        faces_frontal = face_cascade.detectMultiScale(gray, 1.1, 5)
        faces.extend(faces_frontal)

        # Detect right-facing profiles
        faces_right = profile_cascade.detectMultiScale(gray, 1.1, 5)
        faces.extend(faces_right)

        # Detect left-facing profiles (flip horizontally, then detect)
        flipped_gray = cv2.flip(gray, 1)
        faces_left_flipped = profile_cascade.detectMultiScale(flipped_gray, 1.1, 5)
        for (x, y, w, h) in faces_left_flipped:
            x_original = gray.shape[1] - x - w
            faces.append((x_original, y, w, h))

        # Blur each detected face
        for (x, y, w, h) in faces:
            face_region = frame[y:y+h, x:x+w]
            face_region = cv2.GaussianBlur(face_region, (199, 199), 60)
            frame[y:y+h, x:x+w] = face_region

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB", use_column_width=True)

        time.sleep(0.03)

    cap.release()
