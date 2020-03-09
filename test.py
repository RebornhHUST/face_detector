import pickle
import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
labels = {}
with open("label.pickle", "rb") as f:
    labels = pickle.load(f)
for i in range(5):
    a = i + 1
    img = str(a) + '.png'

    image = cv2.imread(img)
    face = cv2.CascadeClassifier("C:\opencv\sources\data\haarcascades\haarcascade_frontalface_alt2.xml")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        gray_image = image[y:y+h, x:x+w]



