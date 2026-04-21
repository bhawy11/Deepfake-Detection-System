import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

import gdown
import os

# MODEL_PATH = "deepfake_model.h5"

# if not os.path.exists(MODEL_PATH):
#     url = "https://drive.google.com/uc?id=12CHoWQzqK7NslhmFsDlBpRAWVPw7mz-M"
#     gdown.download(url, MODEL_PATH, quiet=False)

# Load model
# model = load_model("deepfake_model.h5")

#new load model

import h5py, json

def load_model_compatible(path):
    with h5py.File(path, 'r') as f:
        config = json.loads(f.attrs['model_config'])

    def fix_input_layer(cfg):
        if isinstance(cfg, dict):
            if cfg.get('class_name') == 'InputLayer':
                c = cfg['config']
                if 'batch_shape' in c:
                    c['batch_input_shape'] = c.pop('batch_shape')
                c.pop('optional', None)
            for v in cfg.values():
                fix_input_layer(v)
        elif isinstance(cfg, list):
            for item in cfg:
                fix_input_layer(item)

    fix_input_layer(config)
    from tensorflow.keras.models import model_from_json
    model = model_from_json(json.dumps(config))
    model.load_weights(path)
    return model

model = load_model_compatible("deepfake_model.h5")

#converted_model
# model = load_model("deepfake_model_fixed.keras")

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

IMG_SIZE = 128
THRESHOLD = 0.6

st.title("DeepFake Detection System")

st.warning(
    "⚠️ This system detects face-swap deepfakes. "
    "Pure AI-generated images may not always be detected."
)

uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"]
)

def face_present(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=6, minSize=(60, 60)
    )
    return len(faces) > 0


if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()), dtype=np.uint8
    )
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Unable to read image")
    else:
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # FACE CHECK
        if not face_present(img):
            st.error("No human face detected → Possibly AI-generated image")

        else:
            # Preprocess
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img_resized = img_resized.astype(np.float32) / 255.0
            img_resized = np.expand_dims(img_resized, axis=0)

            # Prediction
            prediction = model.predict(img_resized)[0][0]
            st.write("Prediction score:", float(prediction))

            if prediction >= THRESHOLD:
                st.error("FAKE IMAGE")
            else:
                st.success("REAL IMAGE")
