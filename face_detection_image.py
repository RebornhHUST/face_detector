import cv2
import numpy
import pickle

face = cv2.CascadeClassifier('C:\opencv\sources\data\haarcascades\haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
labels = {}
with open("label.pickle", "rb") as f:
    labels = pickle.load(f)

img = cv2.imread('1.png', )
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2, )
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + h]
    id_, conf = recognizer.predict(roi_gray)
    for label in labels:
        if labels[label] is id_:
            name = label
            color = (255, 255, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, name, (x, y), font, 1, color, 2, cv2.LINE_AA)

cv2.imshow("anh", img)
cv2.waitKey()