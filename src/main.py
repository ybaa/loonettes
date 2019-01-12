from src.data.camera import Camera
from src.constants import STEREO_CAM, STEREO_CAPTURES_FROM_CAM_PATH
from src.data.stereoImagesConverter import StereoImagesConverter
from src.models.cnnCifar import CNNCifar
from src.features.cifarHelper import CifarHelper
from src.visualization.speaker import Speaker
from src.data.cifarLoader import CifarLoader
import cv2
from src.models.divisionDetector import DivisionDetector
from matplotlib import pyplot as plt


# cnn = CNNCifar()
# cnn.run_learning_session()
#
# cifar_helper = cnn.load_and_prepare_set()
# pred = cnn.predict_single_image(cifar_helper.test_images[0])
# str_pred = cifar_helper.batches_meta[b'label_names'][pred[0]].decode("utf-8")
# Speaker.say_recognition(str_pred)

# cam = Camera(STEREO_CAM['LEFT'], STEREO_CAM['RIGHT'])
# frame_left, frame_right = cam.capture_stereo(STEREO_CAPTURES_FROM_CAM_PATH, save=False)

frame_left = cv2.imread('../data/raw/captures/stereo/cam2_154253530281275_L.jpg')
frame_right = cv2.imread('../data/raw/captures/stereo/cam2_154253530281275_R.jpg')

stereo_img = StereoImagesConverter(frame_left, frame_right)
disparity_map = stereo_img.create_disparity_map()

frame_left_appended = stereo_img.append_img_with_disparity_map(frame_left, disparity_map)
frame_right_appended = stereo_img.append_img_with_disparity_map(frame_right, disparity_map)

division_detector = DivisionDetector()
# division_detector.divide(frame_left_appended,1)
division_detector.divide_recursively(2, frame_left_appended)

# cnn_cifar_100 = CNNCifar(labels_amount=100)
# # cnn_cifar_100.run_learning_session(restore=False, save=True)
# cifar_helper = cnn_cifar_100.load_and_prepare_set()
# pred = cnn_cifar_100.predict_single_image(cifar_helper.test_images[0])
# str_pred = cifar_helper.batches_meta[b'fine_label_names'][pred[0]].decode("utf-8")
# Speaker.say_recognition(str_pred)


