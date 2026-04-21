import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split

data=[]
labels=[]

real_path="dataset/real"
fake_path="dataset/fake"

img_size = 128  
max_images_per_class = 2000  

count = 0
for img in os.listdir(real_path):
    if count >= max_images_per_class:
        break
    img_path=os.path.join(real_path,img)
    image=cv2.imread(img_path)
    if image is not None:  # Check if image was loaded successfully
        image=cv2.resize(image,(img_size,img_size))
        data.append(image)
        labels.append(0)
        count += 1
    
count = 0
for img in os.listdir(fake_path):
    if count >= max_images_per_class:
        break
    img_path=os.path.join(fake_path,img)
    image = cv2.imread(img_path)
    if image is not None:  # Check if image was loaded successfully
        image = cv2.resize(image,(img_size,img_size))
        data.append(image)
        labels.append(1)
        count += 1

data=np.array(data, dtype=np.float32)/255.0
labels=np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

'''
*# Free up memory
del data
del labels
import gc
gc.collect()
'''



'''
model=Sequential([
    Conv2D(32,(3,3), activation='relu', input_shape=(img_size,img_size,3)),
    MaxPooling2D(2,2),
    Conv2D(64,(3,3), activation='relu'), 
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
'''

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])



model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
'''
class_weight = {a
    0:1.0,
    1: len(real_images) / len(fake_images)
}
'''
# Fixed parameter name 'epochs' (was 'epoch')
model.fit(x_train, y_train, epochs=20, validation_data=(x_test,y_test), batch_size=32)

model.save("deepfake_model.h5")