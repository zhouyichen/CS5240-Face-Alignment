import numpy as np

def U(r):
	return r * r * np.log10(r + 1e-10)

def solve_tps(input_points, output_points):
	"""
	Input:
		input_points: n * 2 matrix of coordinates of n landmark points
		output_points: n * 2 matrix of coordinates of n landmark points
	Return:
		w: n * 2 matrix
		a: 3 * 2 matrix
	"""
	n = input_points.shape[0]

	v = output_points - input_points

	K = np.zeros((n, n))
	for i in range(n):
		for j in range(n):
			r = np.linalg.norm(input_points[i] - input_points[j])
			K[i, j] = U(r)

	P = np.hstack((input_points, np.ones((n, 1))))
	Pt = P.T

	L = np.zeros((n+3, n+3))
	L[:n, :n] = K
	L[:n, n:] = P
	L[n:, :n] = Pt

	b = np.zeros((n+3, 2))
	b[:n, :] = v

	wa = np.linalg.pinv(L).dot(b)

	w = wa[:n]
	a = wa[n:]

	return w, a

def apply_tps(input_point, input_points, w, a):
	"""
	Apply TPS to one 2d point from the input image according to w and a
	Input:
		input_point: 1 * 2 array containing the coordinates
		input_points: n * 2 matrix of coordinates
	Return:
		destination_coord: 1 * 2 array of transformed coordinates of the input point
	"""
	homo_input_point = np.append(input_point, 1)
	r = np.linalg.norm(input_points - input_point, axis=1)
	weighted_e = (U(r)).dot(w)
	# print weighted_e
	change = homo_input_point.dot(a) + weighted_e
	destination_coord = input_point + change
	return destination_coord
