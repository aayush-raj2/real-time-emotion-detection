import streamlit as st
import cv2
import pandas as pd
from src.predict import predict
from src.utils import log_emotion

st.title("Emotion Dashboard")

run = st.checkbox("Start")
frame_window = st.image([])
chart = st.empty()

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        break

    results = predict(frame)

    for (x,y,w,h,label) in results:
        log_emotion(label)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,label,(x,y-10),0,1,(0,255,0),2)

    frame_window.image(frame, channels="BGR")

    try:
        df = pd.read_csv("logs/emotions.csv", names=["time","emotion"])
        chart.bar_chart(df["emotion"].value_counts())
    except:
        pass
