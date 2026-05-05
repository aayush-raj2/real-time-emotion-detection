from datetime import datetime

def log_emotion(label):
    with open("logs/emotions.csv", "a") as f:
        f.write(f"{datetime.now()},{label}\n")
