import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from common import *

POLY_DEGREE = 2
poly = PolynomialFeatures(degree=POLY_DEGREE)

def evaluate_results(res):
	with open(NUMBER_LM_PRE+LANDMAKRS_DATA) as f:
		# initialize the CSV reader
		data = np.genfromtxt(f, delimiter=",")
		size_of_data = data.shape[0] * NUMBER_OF_POSES

		y = data[:, :NUMBER_OF_LANDMARK_PER_FACE * 2]
		y = np.repeat(y, NUMBER_OF_POSES, axis=0)

		X = data[:, NUMBER_OF_LANDMARK_PER_FACE * 2:].reshape((size_of_data, NUMBER_OF_LANDMARK_PER_FACE * 2))
		poly_X = poly.fit_transform(X)

		predicted_homo_array = res.predict(poly_X)
		homo_matrices = get_homo_matrices(predicted_homo_array)

		X = X.reshape((size_of_data, 2, NUMBER_OF_LANDMARK_PER_FACE))

		extended_X = np.append(X, np.ones((size_of_data, 1, NUMBER_OF_LANDMARK_PER_FACE)), axis = 1)

		error = 0

		for i in range(size_of_data):
			x = extended_X[i]
			homo_mat = homo_matrices[i]
			predicted_y = homo_mat.dot(x)
			predicted_y = (predicted_y[:2] / predicted_y[2]).reshape(2 * NUMBER_OF_LANDMARK_PER_FACE)
			diff = predicted_y - y[i]
			error += np.sum(np.square(diff)) / (2 * NUMBER_OF_LANDMARK_PER_FACE)

		average_error = error / size_of_data
		print average_error
		return average_error


def get_homo_matrices(incomplete_homo_array):
	complete_homo_array = np.append(incomplete_homo_array, np.ones((incomplete_homo_array.shape[0], 1)), axis=1)
	return complete_homo_array.reshape((-1, 3, 3))

def get_homo_matrix(model, landmarks):
	landmarks = poly.fit_transform(landmarks)
	homo_flat = model.predict(landmarks)
	homo = np.zeros(9)
	homo[:8] = homo_flat
	homo[8] = 1
	return homo.reshape((3, 3))

def get_transformed_landmarks(homography, original_landmarks):
	angled_view_landmarks = np.append(original_landmarks, np.ones((NUMBER_OF_LANDMARK_PER_FACE, 1)), axis=1)
	return

def train_based_on_data(data_file, output_file):
	with open(data_file) as f:
		# initialize the CSV reader

		data = np.genfromtxt(f, delimiter=",")
		X = data[:, :NUMBER_OF_LANDMARK_PER_FACE * 2]
		y = data[:, NUMBER_OF_LANDMARK_PER_FACE * 2:]
		X = poly.fit_transform(X)

		reg = linear_model.LinearRegression()
		res = reg.fit(X, y)
		evaluate_results(res)
		joblib.dump(res, output_file, compress=2)
		# close the reader
		f.close()

if __name__ == "__main__":
	train_based_on_data(NUMBER_LM_PRE+HOMOGRAPHY_DATA, 
		NUMBER_LM_PRE+'homo_train_param_' + str(POLY_DEGREE) + '.pkl')
	