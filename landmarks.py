import dlib
import numpy as np
import cv2

face_detector = dlib.get_frontal_face_detector()
PREDICTOR_PATH = "data/shape_predictor_68_face_landmarks.dat"
landmark_predictor = dlib.shape_predictor(PREDICTOR_PATH)

def get_landmarks(img, face_image=False):
	d = face_detector(img, 1)[0]
	pre = landmark_predictor(img, d)
	return np.array([(p.x, p.y) for (i, p) in enumerate(pre.parts())])

def annotate_landmarks(image, landmarks):
	img = image.copy()
	for idx, point in enumerate(landmarks):
		pos = (point[0], point[1])
		cv2.circle(img, pos, 3, (0, 255, 255))
	return img