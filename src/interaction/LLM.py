import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import requests
import json
import speech_recognition as sr
import pyttsx3
import time
from collections import deque

# Load models and initialize components
model = load_model("emotion.h5")
emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

API_KEY = "your-api-key"
API_URL = "https://api.groq.com/"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 220)
tts_engine.setProperty("volume", 0.9)

recognizer = sr.Recognizer()

# Face detection (using Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    st.error("Error: Could not load Haar Cascade classifier. Check OpenCV installation.")
    face_cascade = None


emotion_history = deque(maxlen=5)

# Function to speak
def speak(text):
    print(f"Robot: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()

# Function to listen
def listen():
    with sr.Microphone() as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        st.write("Speak now...")
        try:
            print("Listening for audio...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.WaitTimeoutError:
            speak("No response detected. That’s suspicious.")
            print("Timeout error detected.")
            return "NO_RESPONSE"
        except sr.UnknownValueError:
            speak("Didn’t catch that. Could you say it again?")
            print("Unknown value error.")
            return listen()
        except sr.RequestError as e:
            speak(f"Trouble hearing you: {e}. Let’s try again.")
            print(f"Request error: {e}")
            return ""
        except Exception as e:
            print(f"Microphone error: {e}")
            return ""

def ask_groq(prompt, conversation_history=None):
    if conversation_history is None:
        conversation_history = []
    messages = conversation_history + [{"role": "user", "content": prompt}]
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": messages,
        "max_tokens": 150,
        "temperature": 0.7
    }
    response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        return f"Error: {response.status_code} - {response.text}"


def detect_emotion(frame):
    if frame is None or face_cascade is None:
        return "Neutral"
    
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]  
        face_roi = gray[y:y+h, x:x+w]
        resized_face = cv2.resize(face_roi, (48, 48))
        normalized_face = resized_face / 255.0
        reshaped_face = np.expand_dims(normalized_face, axis=0)
        reshaped_face = np.expand_dims(reshaped_face, axis=-1)
        
        predictions = model.predict(reshaped_face, verbose=0)
        emotion_label = emotion_classes[np.argmax(predictions)]
        emotion_history.append(emotion_label)
        
        
        if len(emotion_history) == emotion_history.maxlen:
            emotion_counts = {e: emotion_history.count(e) for e in set(emotion_history)}
            return max(emotion_counts, key=emotion_counts.get)
        return emotion_label
    return "Neutral"


def main():
    st.title("Emotion-Aware Security Chat")
    st.text("Webcam feed with emotion detection and conversational security check")

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open webcam. Check permissions or connection. Trying index 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            st.error("Error: No webcam available.")
            return

    st.success("Webcam connected successfully!")
    FRAME_WINDOW = st.image([])


    if "running" not in st.session_state:
        st.session_state.running = False


    start_button = st.button("Start Security Check", key="start_button")
    stop_button = st.button("Stop", key="stop_button")

    speak("Security Check: Let’s chat for a bit!")
    conversation = [
        {"role": "system", "content": "You’re a security guard patrolling a neighborhood, asking natural, adaptive questions to determine if someone is suspicious. Adjust your tone based on the user’s facial emotion (e.g., friendly for 'Happy', cautious for 'Angry'). Start with a casual opener."}
    ]
    initial_question = "Hey, what brings you out in the neighborhood this late?"
    speak(initial_question)

    responses = []
    max_turns = 10

    while True:
        if start_button and not st.session_state.running:
            st.session_state.running = True
        if stop_button:
            st.session_state.running = False
            break

        if st.session_state.running:

            ret, frame = cap.read()
            if ret and frame is not None:
                current_emotion = detect_emotion(frame)
                cv2.putText(frame, f'Emotion: {current_emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                st.write(f"Detected Emotion: {current_emotion}")

            user_answer = listen() if st.session_state.running else ""
            if user_answer == "NO_RESPONSE":
                speak("You’re being suspicious with that silence!")
                print("\nSecurity Verdict: Suspicious")
                break
            if user_answer:
                responses.append(user_answer)
                conversation.append({"role": "user", "content": f"{user_answer} (Emotion: {current_emotion})"})
                st.write(f"You said: {user_answer}")

                if len(responses) >= 2:
                    verdict_prompt = (
                        "Here’s the conversation so far:\n" +
                        "\n".join([f"Guard: {c['content']}" if c['role'] == 'assistant' else f"User: {c['content']}" for c in conversation[1:]]) +
                        "\nAs a security guard, decide if this person is suspicious based on evasiveness, contradictions, hostility, or emotional cues. "
                        "If you have enough info, say 'Verdict: Suspicious' or 'Verdict: Not Suspicious' and stop asking. "
                        "If not, ask another natural follow-up question tailored to the latest emotion."
                    )
                    next_step = ask_groq(verdict_prompt, conversation)
                    if "Verdict:" in next_step:
                        if "Suspicious" in next_step:
                            speak("You’re being suspicious!")
                        else:
                            speak("Looks like you’re all good!")
                        print(f"\nSecurity Verdict: {next_step}")
                        break
                    else:
                        speak(next_step)
                        conversation.append({"role": "assistant", "content": next_step})
                        time.sleep(1)  
                else:
                    follow_up_prompt = (
                        "Based on this conversation:\n" +
                        "\n".join([f"Guard: {c['content']}" if c['role'] == 'assistant' else f"User: {c['content']}" for c in conversation[1:]]) +
                        f"\nThe user’s latest emotion is {current_emotion}. Ask a natural, adaptive follow-up question to probe further about their presence in the neighborhood, adjusting tone based on emotion."
                    )
                    next_question = ask_groq(follow_up_prompt, conversation)
                    speak(next_question)
                    conversation.append({"role": "assistant", "content": next_question})
                    time.sleep(1)  

        time.sleep(0.1)  

    if len(responses) == max_turns:
        final_verdict = ask_groq(
            "Conversation:\n" +
            "\n".join([f"Guard: {c['content']}" if c['role'] == 'assistant' else f"User: {c['content']}" for c in conversation[1:]]) +
            "\nIs this person suspicious in the neighborhood based on their responses and emotions? Say 'Verdict: Suspicious' or 'Verdict: Not Suspicious'."
        )
        if "Suspicious" in final_verdict:
            speak("You’re being suspicious!")
        else:
            speak("Looks like you’re all good!")
        print(f"\nSecurity Verdict: {final_verdict}")
    cap.release()

if __name__ == "__main__":
    main()