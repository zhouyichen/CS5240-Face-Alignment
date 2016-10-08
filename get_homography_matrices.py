import cv2
import numpy as np
from common import *

def to_np_mat(input_list):
	return np.array(input_list).reshape((NUMBER_OF_LANDMARK_PER_FACE, 2))

def process_file(data_file, output_file):
	output = open(output_file, "w")
	with open(data_file) as f:
		# initialize the CSV reader

		data = np.genfromtxt(f, delimiter=",")
		for row in data:
			pics = row.reshape((NUMBER_OF_POSES + 1, NUMBER_OF_LANDMARK_PER_FACE * 2))
			frontal_view_landmarks = pics[0].reshape((NUMBER_OF_LANDMARK_PER_FACE, 2))
			for angled_view in pics[1:]:
				angled_view_landmarks = angled_view.reshape((NUMBER_OF_LANDMARK_PER_FACE, 2))
				M, mask = cv2.findHomography(angled_view_landmarks, frontal_view_landmarks)

				output_list = angled_view.tolist() + M.reshape((9)).tolist()[:8]
				output.write("%s\n" % (",".join([str(i) for i in output_list])))

		# close the reader
		f.close()

if __name__ == "__main__":
	process_file(NUMBER_LM_PRE+LANDMAKRS_DATA, NUMBER_LM_PRE+HOMOGRAPHY_DATA)