import cv2
import dlib
from tps import apply_tps, solve_tps
from common import *


predictor = dlib.shape_predictor(PREDICTOR_PATH)

def get_landmarks(im):
    h, w, d = im.shape
    rect = dlib.rectangle(0, 0, w, h)
    return np.array([(p.x, p.y) for p in predictor(im, rect).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0], point[1])
        cv2.circle(im, pos, 3, (0, 255, 255))
    return im

def test_tps(reference_image, target_image):
	reference_landmarks = get_landmarks(reference_image)
	target_landmarks = get_landmarks(target_image)

	w, a = solve_tps(target_landmarks, reference_landmarks)

	height, width, _ = reference_image.shape
	result = np.zeros((height, width, 3))

	for index in np.ndindex(width, height):
		p = apply_tps(index, target_landmarks, w, a)

		p = p.astype(int)

		if p[1] >= height or p[0] >= width or p[1] < 0 or p[0] < 0:
			continue
		result[index[1], index[0]] = reference_image[p[1], p[0]]

	result = result.astype('uint8')

	# result = annotate_landmarks(result, target_landmarks)
	# reference_image = annotate_landmarks(reference_image, reference_landmarks)

	cv2.imshow('winname', np.hstack((reference_image, result)))

	cv2.waitKey(0)

	return

if __name__ == "__main__":
	ref = cv2.imread('038_01_01_130_05.png')
	tar = cv2.imread('038_01_01_051_06.png')
	test_tps(ref, tar)



