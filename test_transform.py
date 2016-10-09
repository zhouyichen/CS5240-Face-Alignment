import numpy as np
import cv2
from sklearn.externals import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from extract_landmarks import get_landmarks
from training_with_homography import get_homo_matrix
import sys
from common import *

POLY_DEGREE = 2
poly = PolynomialFeatures(degree=POLY_DEGREE)
class Tansformer():
	def __init__(self, homo_trained_data):
		self.model = homo_trained_data
		self.homo_coef = homo_trained_data.coef_

	def transform_image(self, image):
		landmarks = get_landmarks(image)
		np_landmarks = np.array(landmarks).reshape((NUMBER_OF_LANDMARK_PER_FACE * 2))
		homography_matrix = get_homo_matrix(self.model, np_landmarks)

		transformed = cv2.warpPerspective(image, homography_matrix, (150, 150))
		print homography_matrix
		cv2.imwrite('transformed.jpg', transformed)


if __name__ == "__main__":
	input_image = sys.argv[1]
	homo_trained_data = joblib.load(NUMBER_LM_PRE + 'homo_train_param_' + str(POLY_DEGREE) + '.pkl')
	t = Tansformer(homo_trained_data)
	input_image = cv2.imread(input_image)
	t.transform_image(input_image)