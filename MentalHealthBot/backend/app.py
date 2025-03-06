from sentiment_analysis import analyze_sentiment
from flask import Flask, request, jsonify, send_file

from openai import OpenAI 

from face_emotion import analyze_facial_emotion
import os 

# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import whisper
import librosa
import numpy as np
import torch
import torchaudio
# from google.cloud import texttospeech
import wave
import io # whisper, wave, io used for sst and tts


app = Flask(__name__)


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# openai.api_key = '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'
# client = OpenAI(api_key="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") 

# sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Initialize LangChain Chat Model
chat_model = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature=0.7, openai_api_key='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')

# loading whisper model
whisper_model = whisper.load_model('base')

# # Memory to store conversation history
# memory = ConversationBufferMemory()

# Improved Memory: Stores summary
memory = ConversationSummaryMemory(
    llm = chat_model,
    memory_key='history',
    return_messages=True
)

def analyze_sentiment_bot(text):
    sentiment_score = analyzer.polarity_scores(text)['compound']
    if sentiment_score >= 0.05:
        return "positive"
    elif sentiment_score <= -0.05:
        return "negative"
    else:
        return "neutral"
    
# function to generate AI powered therapy responses with memory
def generate_therapy_suggestions(user_message, sentiment):
    memory.save_context(
        {"input": user_message},
        {"output": f"The user is feeling {sentiment} and needs support."}
    )

    chat_history = memory.load_memory_variables({}).get("history", [])
    

    messages = [
        SystemMessage(content="You are a helpful AI therapist."),
        # HumanMessage(content=f"User's sentiment: {sentiment}. User said: {user_message}")
        HumanMessage(content=f"User is reflecting on their emotions : {sentiment}. Based on their chat history (mention past messages briefly), why do they feel {sentiment}, user said : {user_message}"),
        AIMessage(content="Previously, the user mentioned: " + str(chat_history))

    ]

    if isinstance(chat_history, list):
        messages.extend(chat_history)
    elif isinstance(chat_history, str): 
        messages.append(AIMessage(content=chat_history))

    print("DEBUG: Chat history format:", type(chat_history), chat_history)
    print(f"Received message: {user_message}")
    print("DEBUG: Messages being sent to chat_model:", messages)

    # response = chat_model(messages)
    response = chat_model.invoke(messages)


    print(f"DEBUG: Response type from chat_model: {type(response)}")
    print(f"DEBUG: Response content: {response}")

    # return response.content
    return str(response.content)


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message", "")
    face_emotion = data.get("emotion", None)

    if face_emotion and face_emotion.lower() != "neutral":
        dominant_emotion = face_emotion  # Use facial emotion as priority
    else:
        if not user_message:
            return jsonify({'error': 'Please enter a message or provide facial input'}), 400
        dominant_emotion = analyze_sentiment_bot(user_message)
    
    # sentiment= analyze_sentiment_bot(user_message)
    therapy_suggestion = generate_therapy_suggestions(user_message, dominant_emotion)

    return jsonify({
            "dominant_emotion": dominant_emotion,
            "therapy_response": therapy_suggestion
        })

# Google Cloud TTS Client
# tts_client = texttospeech.TextToSpeechClient()

def transcribe_audio(file_path):
    result = whisper_model.transcribe(file_path)
    return result["text"]

def detect_emotion(file_path):
    y, sr = librosa.load(file_path)
    energy = np.sum(np.abs(y))
    
    # Simple Emotion Detection (Example)
    if energy > 0.1:
        return "Excited"
    elif energy > 0.05:
        return "Neutral"
    else:
        return "Sad"
    
@app.route('/speech-to-text', methods = ['POST'])
def speech_to_text():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    
    audio_file = request.files["audio"]
    audio_path = "temp_audio.wav"
    audio_file.save(audio_path)

    transcribed_text = transcribe_audio(audio_path)
    emotion = detect_emotion(audio_path)

    return jsonify({"text": transcribed_text, "emotion": emotion})

# @app.route("/text-to-speech", methods=["POST"])
# def text_to_speech():
#     text = request.json.get("text", "")

#     synthesis_input = texttospeech.SynthesisInput(text=text)
#     voice = texttospeech.VoiceSelectionParams(language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
#     audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

#     response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

#     with open("output.mp3", "wb") as out:
#         out.write(response.audio_content)

#     return send_file("output.mp3", mimetype="audio/mpeg")

# down this are api's for prototype
def get_chatbot_response(user_input):
    """Generates a chatbot response based on sentiment analysis."""

    # getting sentiment from sentiment_analysis
    sentiment, confidence = analyze_sentiment(user_input)

    if sentiment == 'Positive':
        response_prompt = f"The user is very happy. Respond in a friendly and exicited way: {user_input}"
    elif sentiment == "Negative":
        response_prompt = f"The user is feeling down. Respond with empathy and support: {user_input}"
    else:
        response_prompt = f"The user is neutral. Respond normally: {user_input}"

    # generating response using chatGPt
    # response = openai.ChatCompletion.create(
    #     model = "gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful and empathetic assistant."},
    #         {"role": "user", "content": response_prompt}
    #     ]
    # )

    # return response["choices"][0]['message']['content']
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="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")

    messages = [
        SystemMessage(content="You are a helpful and empathetic assistant."),
        HumanMessage(content=response_prompt)
    ]
    response = llm.invoke(messages)

    return response

def generate_chatbot_response(user_text, face_emotion):
    """Generates chatbot response based on text sentiment and face emotion."""
    
    sentiment, _ = analyze_sentiment(user_text)
    
    # Determine dominant emotion (priority: face > text)
    dominant_emotion = face_emotion if face_emotion else sentiment

    if dominant_emotion == "happy" or sentiment == "POSITIVE":
        response_prompt = f"The user is happy. Respond in a cheerful way: {user_text}"
    elif dominant_emotion == "sad" or sentiment == "NEGATIVE":
        response_prompt = f"The user is feeling down. Respond with empathy and encouragement: {user_text}"
    elif dominant_emotion == "angry":
        response_prompt = f"The user is angry. Respond calmly and try to de-escalate: {user_text}"
    else:
        response_prompt = f"The user is neutral. Respond normally: {user_text}"

    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful and empathetic assistant."},
    #         {"role": "user", "content": response_prompt}
    #     ]
    # )

    # return response["choices"][0]["message"]["content"]

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="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")

    messages = [
        SystemMessage(content="You are a helpful and empathetic assistant."),
        HumanMessage(content=response_prompt)
    ]
    response = llm.invoke(messages)

    return response

@app.route('/unified-emotion', methods = ['POST'])
def unified_emotion():
    """
    API endpoint to analyze text sentiment, facial emotion, and generate chatbot response.
    """
    user_text = request.form.get("text", "")
    file = request.files.get("file")

    face_emotion = None
    if file:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)
        face_emotion = analyze_facial_emotion(file_path)

    chatbot_response = generate_chatbot_response(user_text, face_emotion)

    return jsonify({
        "text_sentiment": analyze_sentiment(user_text)[0],
        "face_emotion": face_emotion,
        "chatbot_response": chatbot_response
    })


@app.route('/chatbot', methods = ['POST'])
def chatbot():
    """API endpoint for chatbot interactions. """
    data = request.json
    user_input = data.get("text", "")

    if not user_input:
        return jsonify({"error":"No text provided"}), 400
    
    chatbot_response = get_chatbot_response(user_input=user_input)

    return jsonify({"response": chatbot_response})

@app.route("/face-emotion", methods=["POST"])
def face_emotion():
    # Debugging step
    print(request.files.keys())
    """
    API endpoint to analyze facial emotions from an image
    """

    if "files" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['files']
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    emotion = analyze_facial_emotion(file_path)

    return jsonify({"emotion": emotion})

if __name__ == "__main__":
    app.run(debug=True)