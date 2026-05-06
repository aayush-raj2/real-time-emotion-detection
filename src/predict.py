import cv2
import numpy as np
from tensorflow.keras.models import load_model
from config import EMOTIONS
from smooth import EmotionSmoother

model = load_model("models/emotion_model.h5")
smoother = EmotionSmoother()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def predict(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)

    results = []

    for (x,y,w,h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi,(48,48))
        roi = roi.reshape(48,48,1)
        roi = np.repeat(roi, 3, axis=-1)
        roi = roi/255.0
        roi = roi.reshape(1,48,48,3)

        preds = model.predict(roi)
        idx = smoother.smooth(preds)
        label = EMOTIONS[idx]

        results.append((x,y,w,h,label))

    return results
