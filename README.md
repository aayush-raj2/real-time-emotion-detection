# 😄 Real-Time Emotion Detection System

## 📌 Overview

A real-time facial emotion detection system using MobileNetV2 with smoothing, logging, and performance evaluation.

## 🚀 Features

* Deep Learning (MobileNetV2)
* Real-time webcam detection
* Emotion smoothing (reduces flicker)
* Emotion logging (CSV)
* Confusion matrix & accuracy evaluation
* Streamlit dashboard

## ▶️ Run

Install:
pip install -r requirements.txt

Train:
python src/train.py

Run detection:
python -m src.main

Run UI:
streamlit run app/app.py

Evaluate:
python evaluation/evaluate.py

## 📊 Outputs

* logs/emotions.csv
* results/confusion_matrix.png
* results/accuracy_plot.png

## 👨‍💻 Author

Aayush Raj
