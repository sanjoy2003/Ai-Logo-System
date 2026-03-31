import cv2
import numpy as np

def get_brightness_map(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces


def get_saliency_map(image):
    try:
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        success, saliency_map = saliency.computeSaliency(image)

        if not success:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        saliency_map = (saliency_map * 255).astype("uint8")

        # smooth noise
        saliency_map = cv2.GaussianBlur(saliency_map, (7, 7), 0)

        return saliency_map

    except:
        return np.zeros(image.shape[:2], dtype=np.uint8)