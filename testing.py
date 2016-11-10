import cv2
import glob
import numpy as np
import sys
from transform_with_tps import frontalise_with_tps
from landmarks import get_landmarks, annotate_landmarks

frontal_folder = "05_1"
data_folder = 'data/PIE/'
poses_folders = [
	"19_1",
	"19_0",
	"04_1",
	"05_0",
	"14_0",
	"13_0",
	"08_0",
	"08_1",
]
lightings = ["05", "06", "07", "08", "09"]


def filter_landmarks_for_testing(landmarks):
	keypoints = range(17, 68)
	result = []
	for (i, p) in enumerate(landmarks):
		if i in keypoints:
			result.append(p)
	return np.array(result)
	# return landmarks

def calculate_error(oblique_landmarks, frontal_landmarks):
	source = filter_landmarks_for_testing(oblique_landmarks)
	target = filter_landmarks_for_testing(frontal_landmarks)

	'''
	rearrange source points into a 2*n by 4 matrix
		x1, y1,  1, 0
		y1, -x1, 0, 1
		...
	'''
	n_points = source.shape[0]
	source_mat = np.zeros((2 * n_points, 4))
	for i in range(n_points):
		p = source[i]
		source_mat[2*i] = np.array([p[0], p[1], 1, 0])
		source_mat[2*i+1] = np.array([p[1], -p[0], 0, 1])

	'''
	rearrange target points into a 2*n by 1 matrix
		x1
		y1
		...
	'''
	target = target.reshape([2 * n_points, 1])
	sum_squared_error = np.linalg.lstsq(source_mat, target)[1][0]
	mean_squared_error = sum_squared_error / n_points

	return mean_squared_error

def do_testing_on_dataset(input_dataset_path, reference_landmarks):
	errors = np.zeros((len(poses_folders), len(lightings)))
	total_number_of_people = 0
	for person_path in glob.glob(input_dataset_path + "*"):
		person_path += '/01/'
		print person_path
		total_number_of_people += 1
		frontal_faces = {}
		for image_path in glob.glob(person_path + frontal_folder + "/*.png"):
			lighting = image_path[-6:-4]
			if lighting in lightings:
				img = cv2.imread(image_path)
				landmarks = get_landmarks(img, face_image=True)
				frontal_faces[lighting] = landmarks

		for p, pose_path in enumerate(poses_folders):
			pose_folder_path = person_path + pose_path
			for image_path in glob.glob(pose_folder_path + "/*.png"):
				lighting = image_path[-6:-4]
				if lighting in lightings:
					img = cv2.imread(image_path)
					frontalised = frontalise_with_tps(img, reference_landmarks, face_image=True)
					oblique_landmarks = get_landmarks(frontalised, face_image=True)
					frontal_landmarks = frontal_faces[lighting]
					error = calculate_error(oblique_landmarks, frontal_landmarks)
					print error
					errors[p, lightings.index(lighting)] += error
	return errors / total_number_of_people

if __name__ == "__main__":
	reference_landmarks = np.load('average_frontal.npy')
	mean_squared_errors = do_testing_on_dataset(data_folder, reference_landmarks)
	print mean_squared_errors
	np.save('mean_squared_error.npy', mean_squared_errors)

