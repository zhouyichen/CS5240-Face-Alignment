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

		x1 = int(p[0])
		y1 = int(p[1])
		x2 = int(p[0] + 1)
		y2 = int(p[1] + 1)

		# if u not in the source image, just ignore
		if x2 >= width or y2 >= height or x1 < 0 or y1 < 0:
			continue

		# bilinear interpolation
		x = p[0]
		y = p[1]
		xt = x2 - x
		yt = y2 - y
		c1 = xt * reference_image[y1, x1] + (1 - xt) * reference_image[y1, x2]
		c2 = xt * reference_image[y2, x1] + (1 - xt) * reference_image[y2, x2]
		c3 = yt * c1 + (1 - yt) * c2

		result[index[1], index[0]] = c3

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



