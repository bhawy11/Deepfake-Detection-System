import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
model=load_model("deepfake_model.h5")


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def face_present(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return len(faces) > 0


img = cv2.imread(r"C:\Users\Bhawy11\Desktop\DeepFake_Detection_System\Dataset\test\ChatGPT Image Jan 20, 2026, 12_55_21 AM.png")

'''
img = cv2.resize(img, (128,128))
img = img / 255.0
img=np.expand_dims(img,axis=0)

prediction = model.predict(img)[0][0]
'''

THRESHOLD = [0.4, 0.5, 0.6, 0.7]


if face_present(img):
    img = cv2.resize(img, (128,128))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]
    print("prediction score:", prediction)

    if prediction >= THRESHOLD:
        print("FAKE IMAGE") 
    else:
        print("REAL IMAGE")

else:
    print("No face detected → AI image detection ")
    prediction = None

'''
if prediction >= THRESHOLD:
    print("FAKE IMAGE")
else:
    print("REAL IMAGE")
'''