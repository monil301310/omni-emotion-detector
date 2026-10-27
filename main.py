import cv2
from deepface import DeepFace
from transformers import pipeline
import speech_recognition as sr
import warnings
warnings.filterwarnings("ignore")

# Load the NLP model for text and voice
text_model = pipeline("text-classification", model="bhadresh-psavani/distilbert-base-uncased-emotion")

def analyze_text(text):
    result = text_model(text)
    print(f"\n[RESULT] Emotion: {result[0]['label'].upper()} (Confidence: {round(result[0]['score']*100, 2)}%)")

def analyze_image():
    print("\nOpening webcam... Look at the camera.")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        try:
            # DeepFace analyzes the captured frame
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            print(f"\n[RESULT] Facial Emotion: {result[0]['dominant_emotion'].upper()}")
        except:
            print("\n[ERROR] Could not detect a face.")
    else:
        print("\n[ERROR] Webcam not found.")

def analyze_voice():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("\nListening for 5 seconds... Speak now.")
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=5)
            text = r.recognize_google(audio)
            print(f"You said: '{text}'")
            # Pass the transcribed speech to the text model
            analyze_text(text)
        except sr.WaitTimeoutError:
            print("\n[ERROR] No speech detected.")
        except sr.UnknownValueError:
            print("\n[ERROR] Could not understand the audio.")

while True:
    print("\n--- OMNI EMOTION DETECTOR ---")
    print("1. Text Mode")
    print("2. Webcam Photo Mode")
    print("3. Voice Mode (Mic)")
    print("4. Exit")
    
    choice = input("Select a mode (1-4): ")
    
    if choice == '1':
        user_text = input("Enter a sentence: ")
        analyze_text(user_text)
    elif choice == '2':
        analyze_image()
    elif choice == '3':
        analyze_voice()
    elif choice == '4':
        print("Shutting down.")
        break
    else:
        print("Invalid choice. Try again.")