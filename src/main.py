from src.data.camera import Camera
from src.constants import STEREO_CAM, STEREO_CAPTURES_FROM_CAM_PATH
from src.data.stereoImagesConverter import StereoImagesConverter
from src.models.cnnCifar import CNNCifar
from src.features.cifarHelper import CifarHelper
from src.visualization.speaker import Speaker
from src.data.cifarLoader import CifarLoader
import cv2


# cnn = CNNCifar()
# cnn.run_learning_session()
#
# cifar_helper = cnn.load_and_prepare_set()
# pred = cnn.predict_single_image(cifar_helper.test_images[0])
# str_pred = cifar_helper.batches_meta[b'label_names'][pred[0]].decode("utf-8")
# Speaker.say_recognition(str_pred)

# cam = Camera(STEREO_CAM['LEFT'], STEREO_CAM['RIGHT'])
# frame_left, frame_right = cam.capture_stereo(STEREO_CAPTURES_FROM_CAM_PATH, save=False)

# stereo_img = StereoImagesConverter(frame_left, frame_right)
# stereo_img.create_depth_map()

# cnn_cifar_100 = CNNCifar(labels_amount=100)
# # cnn_cifar_100.run_learning_session(restore=False, save=True)
# cifar_helper = cnn_cifar_100.load_and_prepare_set()
# pred = cnn_cifar_100.predict_single_image(cifar_helper.test_images[0])
# str_pred = cifar_helper.batches_meta[b'fine_label_names'][pred[0]].decode("utf-8")
# Speaker.say_recognition(str_pred)


frame_left = cv2.imread('/home/ybaa/Documents/lunettes/data/raw/captures/stereo/cam2_1547067275510002_L.jpg')
frame_right = cv2.imread('/home/ybaa/Documents/lunettes/data/raw/captures/stereo/cam2_1547067275510002_R.jpg')
stereo_img = StereoImagesConverter(frame_right, frame_left)
stereo_img.create_depth_map()
# stereo_img.test_parameters()