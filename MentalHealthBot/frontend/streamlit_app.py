import streamlit as st
import requests
import tempfile
import av
import numpy as np
import librosa
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import base64

# Flask API URL
API_URL = "http://127.0.0.1:5000"

# Streamlit Page Config
st.set_page_config(page_title="AI Mental Health Chatbot", page_icon="🧠")

# Streamlit UI
st.title("🧠 AI Mental Health Chatbot")
st.write("🌟 **Your AI Support System for Mental Well-being.** Type your concerns or speak out.")

# Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 📌 **1. TEXT CHATBOT**
user_message = st.text_input("💬 Your Message", "")

if st.button("🚀 Get Response"):
    if user_message:
        response = requests.post(f"{API_URL}/chat", json={"message": user_message})
        if response.status_code == 200:
            bot_reply_sentiment = response.json()["dominant_emotion"]
            bot_reply_therapy_suggestion = response.json()["therapy_response"]

            st.session_state.chat_history.append(f"**You:** {user_message}")
            st.session_state.chat_history.append(f"🤖 **Chatbot:** {bot_reply_therapy_suggestion} (Emotion: {bot_reply_sentiment})")

            # 🔊 Text-to-Speech (TTS)
            tts_response = requests.post(f"{API_URL}/text-to-speech", json={"text": bot_reply_therapy_suggestion})
            if tts_response.status_code == 200:
                audio_data = tts_response.content
                audio_base64 = base64.b64encode(audio_data).decode()
                st.audio(audio_data, format="audio/mp3")
        else:
            st.session_state.chat_history.append("❌ Error: Could not get response.")

# Display chat history
for message in st.session_state.chat_history:
    st.write(message)

# 📌 **2. VOICE RECORDING & SPEECH ANALYSIS**
st.write("🎙️ **Speak Your Concerns**")

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_buffer = []

    def recv(self, frame: av.AudioFrame):
        audio_array = np.array(frame.to_ndarray())
        self.audio_buffer.append(audio_array)
        return frame

webrtc_ctx = webrtc_streamer(
    key="speech-recorder",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": False, "audio": True},
)

if webrtc_ctx.audio_receiver:
    audio_processor = AudioProcessor()
    webrtc_ctx.audio_receiver.add_processor(audio_processor)

    if st.button("🔴 Stop & Analyze"):
        # Save recorded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            librosa.output.write_wav(temp_audio.name, np.concatenate(audio_processor.audio_buffer), sr=16000)
            temp_audio_path = temp_audio.name

        # Send audio to backend for Speech-to-Text & Emotion Detection
        with open(temp_audio_path, "rb") as f:
            files = {"audio": f}
            response = requests.post(f"{API_URL}/speech-to-text", files=files)

        if response.status_code == 200:
            transcribed_text = response.json()["text"]
            detected_emotion = response.json()["emotion"]

            st.write(f"**Transcribed Text:** {transcribed_text}")
            st.write(f"🧠 **Detected Emotion:** {detected_emotion}")

            # Get therapy response
            response_therapy = requests.post(f"{API_URL}/chat", json={"message": transcribed_text})
            if response_therapy.status_code == 200:
                therapy_response = response_therapy.json()["therapy_response"]
                st.write(f"🤖 **Chatbot:** {therapy_response}")

                # 🔊 Text-to-Speech (TTS)
                tts_response = requests.post(f"{API_URL}/text-to-speech", json={"text": therapy_response})
                if tts_response.status_code == 200:
                    audio_data = tts_response.content
                    st.audio(audio_data, format="audio/mp3")
            else:
                st.write("❌ Error: Unable to get therapy response.")
        else:
            st.write("❌ Error: Unable to process audio.")
