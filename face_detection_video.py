# sử dụng video
import cv2
import pickle

face = cv2.CascadeClassifier('C:\opencv\sources\data\haarcascades\haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
labels = {}
with open("label.pickle", "rb") as f:
    labels = pickle.load(f)

cap = cv2.VideoCapture(0)
i = 0
while True:
    ret, frame = cap.read()
    name = str(i + 1) + '.png'
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2, )
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + h]
        id_, conf = recognizer.predict(roi_gray)
        if 45 <= conf:
            for label in labels:
                if labels[label] is id_:
                    name = label
                    color = (255, 255, 0)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, name, (x, y), font, 1, color, 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
