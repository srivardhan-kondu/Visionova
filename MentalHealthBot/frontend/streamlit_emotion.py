import cv2
import streamlit as st
import requests
import numpy as np
from backend.face_emotion import analyze_facial_emotion  # Your function

CHATBOT_API_URL = "http://127.0.0.1:5000/unified-emotion"

st.set_page_config(page_title="AI Mental Health Chatbot", page_icon="🧠")
st.title("🧠 AI Mental Health Chatbot with Live Emotion Detection")

video_capture = cv2.VideoCapture(0)
frame_placeholder = st.empty()
emotion_placeholder = st.empty()
chat_placeholder = st.empty()

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        st.error("Failed to grab frame.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detected_emotion = analyze_facial_emotion(frame_rgb)

    frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
    emotion_placeholder.write(f"🎭 **Detected Emotion:** {detected_emotion}")

    response = requests.post(CHATBOT_API_URL, json={"text": "", "emotion": detected_emotion})
    chatbot_reply = response.json().get("chatbot_response", "I'm here to help. How are you feeling?")

    chat_placeholder.write(f"💬 **Chatbot:** {chatbot_reply}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
