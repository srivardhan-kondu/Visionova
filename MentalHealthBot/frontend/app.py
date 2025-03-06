import streamlit as st 
import requests

BACKEND_URL = "http://127.0.0.1:5000/unified-emotion"

st.set_page_config(page_title='AI Therapy Bot', layout='centered')

st.title("🧠 AI-Powered Mental Health Therapy Bot")

user_text = st.text_area("📝 Enter your message: ")

uploaded_file = st.file_uploader("📷 Upload an image of your face (optional)", type=["jpg", "png", "jpeg"])

if st.button("Analyze & Get Response"):
    if not user_text and not uploaded_file:
        st.warning("⚠️ Please enter text or upload an image.")
    else:
        # files = {"file": uploaded_file.getvalue()} if uploaded_file else None
        if uploaded_file:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")}
        else:
            files = None

        data = {"text": user_text}

        with st.spinner("Analyzing Emotions...."):
            response = requests.post(BACKEND_URL, files=files, data=data)
            if response.status_code == 200:
                result = response.json()

                st.subheader("🧠 Analysis Results")
                st.write(f"**📝 Text Sentiment:** {result.get('text_sentiment', 'N/A')}")
                st.write(f"**📷 Facial Emotion:** {result.get('face_emotion', 'N/A')}")
                
                st.subheader("🤖 AI Chatbot Response")
                st.success(result.get("chatbot_response", "No response generated."))
            else:
                st.error("❌ Error in processing request. Please try again.")