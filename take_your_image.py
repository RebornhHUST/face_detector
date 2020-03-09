import cv2


face = cv2.CascadeClassifier('C:\opencv\sources\data\haarcascades\haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)
i = 0
while True:
    ret, frame = cap.read()
    name = str(i)+".png"
    i += 1
    cv2.imwrite(name, frame)
    if i == 100:
        exit()

