import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

data = []
labels = []

real_path = "dataset/real"
fake_path = "dataset/fake"

# Load REAL images
for img in os.listdir(real_path):
    img_path = os.path.join(real_path, img)
    image = cv2.imread(img_path)
    image = cv2.resize(image, (224, 224))
    data.append(image)
    labels.append(0)

# Load FAKE images
for img in os.listdir(fake_path):
    img_path = os.path.join(fake_path, img)
    image = cv2.imread(img_path)
    image = cv2.resize(image, (224, 224))
    data.append(image)
    labels.append(1)

data = np.array(data) / 255.0
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

model=Sequential([
    Conv2D(32,(3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64,(3,3), activation='relu'), 
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train,y_train, epoch=5, validation_data=(x_test,y_test))

model.save("deepfake_model.h5")