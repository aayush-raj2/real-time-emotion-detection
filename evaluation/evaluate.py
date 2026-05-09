import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model

data = pd.read_csv("data/fer2013.csv")

faces = []
for p in data['pixels']:
    face = np.array(p.split(' '), dtype='float32').reshape(48,48,1)
    face = np.repeat(face,3,axis=-1)
    faces.append(face)

faces = np.array(faces)/255.0
labels = data['emotion']

model = load_model("models/emotion_model.h5")

preds = model.predict(faces)
y_pred = np.argmax(preds,axis=1)

acc = accuracy_score(labels,y_pred)
print("Accuracy:",acc)

cm = confusion_matrix(labels,y_pred)

plt.imshow(cm)
plt.savefig("results/confusion_matrix.png")

plt.figure()
plt.plot([acc])
plt.savefig("results/accuracy_plot.png")
