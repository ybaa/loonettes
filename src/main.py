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

# cnn = CNNCifar()
# cnn.run_learning_session()
#
# cifar_helper = cnn.load_and_prepare_set()
# pred = cnn.predict_single_image(cifar_helper.test_images[0])
# str_pred = cifar_helper.batches_meta[b'label_names'][pred[0]].decode("utf-8")
# Speaker.say_recognition(str_pred)

# cam = Camera()
# cam.capture_stereo(save=True)
# for i in range(2):
#     frame_left, frame_right = cam.capture_stereo(STEREO_CAPTURES_FROM_CAM_PATH, save=True)
#     print('photo taken')
#     time.sleep(2)


# frame_left = cv2.imread('../data/raw/captures/stereo/cam2_154253530281275_L.jpg')
# frame_right = cv2.imread('../data/raw/captures/stereo/cam2_154253530281275_R.jpg')
#
# stereo_img = StereoImagesConverter(frame_left, frame_right)
# disparity_map = stereo_img.create_disparity_map(save=True)
#
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
# my_cnn.run_learning_session(save=True)
pred_val = my_cnn.predict_single_image(helper.test_images[0])
pred_val_str = helper.le.inverse_transform([pred_val[0]])
print(pred_val_str[0])
# Speaker.say_recognition(pred_val_str[0])

