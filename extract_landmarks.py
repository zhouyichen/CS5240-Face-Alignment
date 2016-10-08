import cv2
import dlib
import glob
from common import *

PREDICTOR_PATH = "data/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)


frontal_folder = "05_1"

poses_folders = [
    "04_1",
    "05_0",
    "08_0",
    "08_1",
    "13_0",
    "14_0",
    "19_0",
    "19_1"
]

data_folder = 'data/PIE/'

def get_landmarks(im):
    width, height, rgb = im.shape
    rect = dlib.rectangle(0, 0, height, width)
    pre = predictor(im, rect)
    key_points_coord = [[p.x, p.y] for (i, p) in enumerate(pre.parts()) if i in key_points]
    return key_points_coord

def covnert_2d_list_to_string(l):
    return  ",".join([str(item) for sublist in l for item in sublist])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        if idx in key_points:
            pos = (point[0, 0], point[0, 1])
            cv2.putText(im, str(idx), pos,
                        fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        fontScale=0.4,
                        color=(0, 0, 255))
            cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def save_landmarks_to_file(input_path, output_file):
    output = open(output_file, "w")
    for person_path in glob.glob(input_path + "*"):
        person_path += '/01/'
        print person_path
        d = {}
        for image_path in glob.glob(person_path + frontal_folder + "/*.png"):
            im = cv2.imread(image_path)
            landmarks = covnert_2d_list_to_string(get_landmarks(im))
            image_number = image_path[-6:-4]
            d[image_number] = landmarks

        for pose_path in poses_folders:
            pose_folder_path = person_path + pose_path
            for image_path in glob.glob(pose_folder_path + "/*.png"):
                im = cv2.imread(image_path)
                landmarks = covnert_2d_list_to_string(get_landmarks(im))
                image_number = image_path[-6:-4]
                d[image_number] += ',' + landmarks
        for key, value in d.iteritems():
            output.write("%s\n" % (value))

if __name__ == "__main__":
    save_landmarks_to_file(data_folder, NUMBER_LM_PRE + LANDMAKRS_DATA)
