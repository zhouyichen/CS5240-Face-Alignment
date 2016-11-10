import cv2
from tps import apply_tps, solve_tps
from landmarks import get_landmarks, annotate_landmarks
import numpy as np
import sys

'''
Example: python transform_with_tps.py path_to_image.png
'''

AVRAGE_FACE = 'symm_average_face.jpg'
AVERAGE_FACE_SIZE = 200.0


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

def draw_convex_hull(img, points, color):
	points = cv2.convexHull(points.astype(int))
	cv2.fillConvexPoly(img, points, color=color)

def create_mask(height, width, landmarks):
	# include face only
	face_blur = 11
	face_mask = np.zeros((height, width))

	face_indexes = list(range(0, 17))

	face_landmarks = landmarks[face_indexes]

	p1 = 2 * landmarks[17] - landmarks[1]
	p2 = 2 * landmarks[26] - landmarks[15]
	face_landmarks = np.append(face_landmarks, [p1, p2], axis=0)

	draw_convex_hull(face_mask, face_landmarks, 1)

	face_mask = np.array([face_mask, face_mask, face_mask]).transpose((1, 2, 0))

	# face_mask = (cv2.GaussianBlur(face_mask, (face_blur, face_blur), 0) > 0) * 1.0
	face_mask = cv2.GaussianBlur(face_mask, (face_blur, face_blur), 0)

	# exclude eyes when do symmetry
	eyes_blur = 11
	eyes_mask = np.zeros((height, width))

	left_eye_indexes = list(range(42, 48))
	right_eye_indexes = list(range(36, 42))

	draw_convex_hull(eyes_mask, landmarks[left_eye_indexes], 1)
	draw_convex_hull(eyes_mask, landmarks[right_eye_indexes], 1)

	eyes_mask = np.array([eyes_mask, eyes_mask, eyes_mask]).transpose((1, 2, 0))

	eyes_mask = (cv2.GaussianBlur(eyes_mask, (eyes_blur, eyes_blur), 0) > 0) * 1.0
	eyes_mask = cv2.GaussianBlur(eyes_mask, (eyes_blur, eyes_blur), 0)

	return face_mask - eyes_mask

def filter_for_tps(landmarks, average=True):
	result = []
	keypoints = ()
	if average:
		keypoints = (0, 2, 4, 8, 12, 14, 16, 17, 26, 27, 33, 36, 39, 42, 45)
	else:
		keypoints = (8, 27, 36, 45)
	for (i, p) in enumerate(landmarks):
		if i in keypoints:
			result.append(p)
	return np.array(result)

def frontalise_with_tps(target_image, reference_landmarks, face_image=False, show_result=False):
	target_landmarks = get_landmarks(target_image, face_image)

	left_face = target_landmarks[1]
	mid = target_landmarks[28]
	right_face = target_landmarks[15]

	left_dist = np.linalg.norm(left_face - mid)
	right_dist = np.linalg.norm(right_face - mid)

	height, width, _ = target_image.shape
	reference_landmarks = reference_landmarks * (min(height, width) / AVERAGE_FACE_SIZE)
	reference_landmarks[:, 1] += height / 2 - reference_landmarks[33][1]
	reference_landmarks[:, 0] += width / 2 - reference_landmarks[30][0]

	r = right_dist / left_dist
	if abs(r - 1) < 0.2:
		reference_for_tps = filter_for_tps(reference_landmarks, average=False)
		target_for_tps = filter_for_tps(target_landmarks, average=False)
	else:
		reference_for_tps = filter_for_tps(reference_landmarks)
		target_for_tps = filter_for_tps(target_landmarks)
	
	r = r ** 1.2
	# print reference_landmarks

	w, a = solve_tps(reference_for_tps, target_for_tps)

	if width % 2:
		width -= 1

	frontalised = np.zeros((height, width, 3))
	for index in np.ndindex(width, height):
		p = apply_tps(index, reference_for_tps, w, a)
		color = bilinear_interpolate(p, target_image)
		frontalised[index[1], index[0]] = color
	
	symmetric = np.zeros((height, width, 3))
	if r < 1: # add left to right
		symmetric[:,width/2:] = np.fliplr(frontalised[:,:width/2]) * (1 - r) + frontalised[:,width/2:] * r
		symmetric[:,:width/2] = frontalised[:,:width/2]
	else: # add right to left
		r = 1 / r
		symmetric[:,:width/2] = np.fliplr(frontalised[:,width/2:]) * (1 - r) + frontalised[:,:width/2] * r
		symmetric[:,width/2:] = frontalised[:,width/2:]

	face_mask = create_mask(height, width, reference_landmarks)
	final_result = face_mask * symmetric + (1 - face_mask) * frontalised

	if show_result:
		# target_image = annotate_landmarks(target_image, target_landmarks)
		final_result = annotate_landmarks(final_result, reference_landmarks.astype(int))

		cv2.imshow('winname', np.hstack((target_image, final_result.astype('uint8'))))
		cv2.waitKey(0)

	return final_result.astype('uint8')

if __name__ == "__main__":
	target = cv2.imread(sys.argv[1])
	try:
		reference_landmarks = np.load('average_frontal.npy')
	except Exception:
		ref = cv2.imread(AVRAGE_FACE)
		reference_landmarks = get_landmarks(ref)
		np.save('average_frontal.npy', reference_landmarks)
	frontalised = frontalise_with_tps(target, reference_landmarks, face_image=False, show_result=True)

