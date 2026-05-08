import cv2
from predict import predict
from utils import log_emotion

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = predict(frame)

    for (x,y,w,h,label) in results:
        log_emotion(label)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,label,(x,y-10),0,1,(0,255,0),2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()
