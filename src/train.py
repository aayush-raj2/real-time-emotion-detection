import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from model import build_model

data = pd.read_csv("data/fer2013.csv")

faces = []
for p in data['pixels']:
    face = np.array(p.split(' '), dtype='float32').reshape(48,48,1)
    face = np.repeat(face, 3, axis=-1)
    faces.append(face)

faces = np.array(faces) / 255.0
labels = to_categorical(data['emotion'], 7)

X_train, X_test, y_train, y_test = train_test_split(faces, labels, test_size=0.2)

model = build_model()
model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=20)

model.save("models/emotion_model.h5")
