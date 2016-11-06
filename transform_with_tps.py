import cv2
import dlib
from tps import apply_tps, solve_tps
import numpy as np

AVRAGE_FACE = 'symm_average_face.jpg'
AVERAGE_FACE_SIZE = 200.0

PREDICTOR_PATH = "data/shape_predictor_68_face_landmarks.dat"
landmark_predictor = dlib.shape_predictor(PREDICTOR_PATH)
face_detector = dlib.get_frontal_face_detector()

def get_landmarks(img):
	d = face_detector(img, 1)[0]
	pre = landmark_predictor(img, d)
	return np.array([(p.x, p.y) for (i, p) in enumerate(pre.parts())])

def annotate_landmarks(image, landmarks):
	img = image.copy()
	for idx, point in enumerate(landmarks):
		pos = (point[0], point[1])
		cv2.circle(img, pos, 3, (0, 255, 255))
	return img

def bilinear_interpolate(p, image):
	height, width, _ = image.shape

	x1 = int(p[0])
	y1 = int(p[1])
	x2 = int(p[0] + 1)
	y2 = int(p[1] + 1)

	# clip the value to fall inside the width and height
	x1 = max(0, min(x1, width - 1))
	x2 = max(0, min(x2, width - 1))
	y1 = max(0, min(y1, height - 1))
	y2 = max(0, min(y2, height - 1))
		
	# bilinear interpolation
	x = p[0]
	y = p[1]
	xt = x2 - x
	yt = y2 - y
	c1 = xt * image[y1, x1] + (1 - xt) * image[y1, x2]
	c2 = xt * image[y2, x1] + (1 - xt) * image[y2, x2]
	return yt * c1 + (1 - yt) * c2

def frontalise_with_tps(reference_image, target_landmarks):
	reference_landmarks = get_landmarks(reference_image)
	height, width, _ = reference_image.shape
	target_landmarks = target_landmarks * (min(height, width) / AVERAGE_FACE_SIZE)

	# print target_landmarks
	w, a = solve_tps(target_landmarks, reference_landmarks)

	result = np.zeros((height, width, 3))
	for index in np.ndindex(width, height):
		p = apply_tps(index, target_landmarks, w, a)
		color = bilinear_interpolate(p, reference_image)
		result[index[1], index[0]] = color
	
	# result = annotate_landmarks(result, target_landmarks.astype(int))
	# reference_image = annotate_landmarks(reference_image, reference_landmarks)
	return result.astype('uint8')

if __name__ == "__main__":
	ref = cv2.imread('testing/Screen Shot 2016-11-06 at 16.46.51.png')
	tar = cv2.imread(AVRAGE_FACE)
	target_landmarks = get_landmarks(tar)
	frontalised = frontalise_with_tps(ref, target_landmarks)

	cv2.imshow('winname', np.hstack((ref, frontalised)))
	cv2.waitKey(0)

