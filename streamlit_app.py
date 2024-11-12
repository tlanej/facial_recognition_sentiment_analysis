import os
import urllib.request
import sys
import streamlit as st
import cv2
import numpy as np
from fer import FER  # Using FER library for faster emotion detection
import subprocess

# Specify the URL of the .dat file and the file name
DATA_FILE_URL = "https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat"
DATA_FILE_NAME = "shape_predictor_68_face_landmarks.dat"

# Function to add CMake to PATH if not already present
def add_cmake_to_path():
    try:
        # Attempt to find CMake using 'which'
        result = subprocess.run(["which", "cmake"], capture_output=True, text=True, check=True)
        cmake_path = result.stdout.strip()
        if cmake_path and os.path.exists(cmake_path):
            os.environ["PATH"] += os.pathsep + os.path.dirname(cmake_path)
            print(f"Added CMake to PATH: {cmake_path}")
        else:
            print("CMake executable not found. Please install CMake using 'sudo apt install cmake'.")
            sys.exit(1)
    except subprocess.CalledProcessError:
        print("CMake not found. Please install CMake using 'sudo apt install cmake'.")
        sys.exit(1)

# Function to check if the .dat file is present and download if missing
def check_and_download_file():
    if not os.path.exists(DATA_FILE_NAME):
        print(f"{DATA_FILE_NAME} not found. Downloading from {DATA_FILE_URL}...")
        try:
            urllib.request.urlretrieve(DATA_FILE_URL, DATA_FILE_NAME)
            print(f"Downloaded {DATA_FILE_NAME} successfully.")
        except Exception as e:
            print(f"Failed to download {DATA_FILE_NAME}. Error: {e}")
            sys.exit(1)
    else:
        print(f"{DATA_FILE_NAME} is already present.")

# Load the pre-trained Haar Cascade for face detection
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

# Initialize the FER model
emotion_detector = FER(mtcnn=True)

# Streamlit app
def main():
    # Step 1: Add CMake to PATH if not already present
    add_cmake_to_path()

    # Step 2: Check if the .dat file is available, if not download it
    check_and_download_file()

    st.title("Facial Recognition and Sentiment Analysis App")
    st.write("This app uses your webcam to detect facial features and analyze sentiment based on your expressions using a lightweight model.")

    # Access webcam feed
    run = st.checkbox('Run Facial Recognition')
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    frame_counter = 0  # Counter to process emotion detection at intervals

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Could not access the webcam.")
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            st.write("No faces detected. Please ensure your face is clearly visible in the webcam.")
        else:
            for (x, y, w, h) in faces:
                # Extract the face region of interest (ROI)
                face_frame = frame[y:y+h, x:x+w]

                # Only analyze sentiment every 15 frames to reduce load
                if frame_counter % 15 == 0:
                    try:
                        # Analyze sentiment based on the face ROI
                        emotion_analysis = emotion_detector.detect_emotions(face_frame)
                        if emotion_analysis:
                            sentiment = emotion_analysis[0]['emotions']
                            dominant_emotion = max(sentiment, key=sentiment.get)
                        else:
                            dominant_emotion = "Could not determine emotion"
                    except Exception as e:
                        dominant_emotion = f"Error: {str(e)}"

                # Draw rectangle around the face and display sentiment
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_counter += 1

    cap.release()

if __name__ == "__main__":
    main()
