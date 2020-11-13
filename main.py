import cv2
from utils.face_recognizer import FaceRecognizer

import requests as rq
from threading import Thread

recognizer = FaceRecognizer()
frame_thickness = 4

video = cv2.VideoCapture(1)
recognizer.load_encodings('faces')


def open_door():
    rq.get('http://192.168.43.251', data={'26': 'off'}, timeout=1)


while True:
    ret, image = video.read()
    image = cv2.resize(image, None, fx=0.5, fy=0.5)

    results, image = recognizer.recognize(image=image, frame_thickness=frame_thickness)
    if len(results) > 0:
        t = Thread(target=open_door)
        t.start()

    cv2.imshow("fr", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
