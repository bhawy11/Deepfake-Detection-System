# Deepfake-Detection-System
Ml based web application that detects whether an image is **Real** or a **DeepFake** using a custom trained CNN model.
As it detects faces so make sure to upload an image with a face in it
## Live Demo
[Click here to try the app] (https://deepfake-detection-system-9n4ccapvwkalfzqqfzua3o.streamlit.app/)
## Features
- Upload any face image and instantly detect if it's real or fake
- Built-in face detection — rejects non-face images automatically
- Confidence score displayed for every prediction
- Simple and clean web interface
- ## Tech Stack
- **Python**
- **TensorFlow / Keras** — CNN model for deepfake detection
- **OpenCV** — Face detection using Haar Cascade
- **Streamlit** — Web app interface
## MODEL FILE
Download here: https://drive.google.com/file/d/12CHoWQzqK7NslhmFsDlBpRAWVPw7mz-M/view?usp=drive_link
## How It Works
1. User uploads an image
2. OpenCV checks if a human face is present
3. If face found → CNN model predicts REAL or FAKE
4. Confidence score is shown along with the result
