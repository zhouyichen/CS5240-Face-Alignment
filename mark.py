import cv2
import dlib
import numpy
import sys

PREDICTOR_PATH = "data/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# #This is using the Dlib Face Detector . Better result more time taking
# def get_landmarks(im):
#     rects = detector(im, 1)
#     rect=rects[0]
#     print type(rect.width())
#     fwd=int(rect.width())
#     if len(rects) == 0:
#         return None,None

#     return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()]),fwd

def get_landmarks(im):
    w, h, d = im.shape
    rect = dlib.rectangle(0,0,h, w)
    return numpy.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        # cv2.putText(im, str(idx), pos,
        #             fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        #             fontScale=0.4,
        #             color=(0, 0, 255))
        cv2.circle(im, pos, 3, (0, 255, 255))
    return im


if __name__ == "__main__":
    input_image = sys.argv[1]
    im=cv2.imread(input_image)
    cv2.imwrite('marked_' + input_image,annotate_landmarks(im,get_landmarks(im)))
