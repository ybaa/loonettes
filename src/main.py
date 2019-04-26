# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # turn off warnings
from src.data.camera import Camera
from src.constants import STEREO_CAM
from src.data.stereoImagesConverter import StereoImagesConverter
from src.models.cnnCifar import CNNCifar
from src.features.cifarHelper import CifarHelper
from src.visualization.speaker import Speaker
from src.data.cifarLoader import CifarLoader
import cv2
from src.models.divisionDetector import DivisionDetector
from matplotlib import pyplot as plt
from src.data.myDatasetLoader import MyDatasetLoader
from src.features.myDatasetHelper import MyDatasetHelper
from src.models.cnnMyDataset import CNNMyDataset
import time
from src.features.myDatasetHelper import MyDatasetHelper
from src.experiments.myDatasetAndDivisionDetector import MyDatasetAndDivisionDetectorManager


# exp1 = MyDatasetAndDivisionDetectorManager()
# exp1.test_for_3_channels()

# cnn = CNNCifar()
# cnn.run_learning_session()
#
# cifar_helper = cnn.load_and_prepare_set()
# pred = cnn.predict_single_image(cifar_helper.test_images[0])
# str_pred = cifar_helper.batches_meta[b'label_names'][pred[0]].decode("utf-8")
# Speaker.say_recognition(str_pred)

# cam = Camera()
# cam.show_constant_mono_capture(rectify=True)
# cam.capture_stereo(save=True)
# cam.capture_constant_stereo_simultaneously()
# cam.calibrate()
# img = cv2.imread('../data/raw/cameraCalibration/cam2_1554708578437131_L.jpg')
# img = cv2.imread('../data/raw/captures/stereo/cam2_154807559582542_R.jpg')
# cam.rectify_img(img)

# frame_left, frame_right = cam.capture_stereo(save=False, rectify=False)
# cv2.imshow('aaa',frame_left)
# cv2.waitKey()

# for i in range(10):
#     frame_left, frame_right = cam.capture_stereo(save=True, rectify=True)
#     Speaker.say_recognition(i)
#     cv2.imshow(str(i), frame_left)
#     cv2.waitKey()
#     cv2.destroyAllWindows()
#     # time.sleep(1)


# frame_left = cv2.imread('../data/raw/captures/stereo/cam2_1548075816280474_L.jpg')
# frame_right = cv2.imread('../data/raw/captures/stereo/cam2_1548075816280474_R.jpg')

# frame_left = cam.rectify_img(frame_left)
# frame_right = cam.rectify_img(frame_right)

# frame_left, frame_right = cam.capture_stereo(save=False)
# cv2.imshow('left', frame_left)
# cv2.imshow('right', frame_right)
# # cv2.waitKey()
# stereo_img = StereoImagesConverter(frame_left, frame_right)
# disparity_map = stereo_img.create_disparity_map(save=False)
# # stereo_img.test_parameters()
#
# stereo_img.calculate_distance(disparity_map)

# frame_left = cv2.imread('../data/raw/captures/stereo/cam2_1542535287840006_L.jpg')
# frame_right = cv2.imread('../data/raw/captures/stereo/cam2_1542535287840006_R.jpg')
#
# stereo_img = StereoImagesConverter(frame_left, frame_right)
# stereo_img.rectify_image(frame_left)
# disparity_map = stereo_img.create_disparity_map(save=False)

# frame_left_appended = stereo_img.append_img_with_disparity_map(frame_left, disparity_map)
# frame_right_appended = stereo_img.append_img_with_disparity_map(frame_right, disparity_map)
#
# division_detector = DivisionDetector()
# # division_detector.divide(frame_left_appended,1)
# division_detector.divide_recursively(2, frame_left_appended)

# cnn_cifar_100 = CNNCifar(labels_amount=100)
# # cnn_cifar_100.run_learning_session(restore=False, save=True)
# cifar_helper = cnn_cifar_100.load_and_prepare_set()
# pred = cnn_cifar_100.predict_single_image(cifar_helper.test_images[0])
# str_pred = cifar_helper.batches_meta[b'fine_label_names'][pred[0]].decode("utf-8")
# Speaker.say_recognition(str_pred)

# my_dataset_loader = MyDatasetLoader()
# my_dataset_loader.pickle_data()
# loaded = my_dataset_loader.load_dataset()

# dataset_appended_with_dm = MyDatasetHelper.crete_disparity_maps_serial(loaded[0][0])
# division_detector = DivisionDetector()

# single_img = loaded[0][0]
#
# division_detector.divided_image.append(single_img)
# division_detector.divide_recursively(2, single_img)
#
# single_img_divided_rescaled = MyDatasetHelper.resize_images(division_detector.divided_image)
#
#
my_cnn = CNNMyDataset()
helper = my_cnn.load_and_prepare_set()
# my_cnn.run_learning_session(save=False)
# pred_val = my_cnn.predict_single_image(helper.test_images[0])
# pred_val_str = helper.le.inverse_transform([pred_val[0]])
# print(pred_val_str[0])
# Speaker.say_recognition(pred_val_str[0])

