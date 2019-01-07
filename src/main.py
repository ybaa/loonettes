from src.data.camera import Camera
from src.constants import STEREO_CAM, STEREO_CAPTURES_FROM_CAM_PATH
from src.data.stereoImagesConverter import StereoImagesConverter
from src.models.cnnCifar import CNNCifar
from src.features.cifarHelper import CifarHelper
from src.visualization.speaker import Speaker
from src.data.cifarLoader import CifarLoader


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

cnn_cifar_100 = CNNCifar(labels_amount=100)
# cnn_cifar_100.run_learning_session(restore=False, save=True)
cifar_helper = cnn_cifar_100.load_and_prepare_set()
pred = cnn_cifar_100.predict_single_image(cifar_helper.test_images[0])
str_pred = cifar_helper.batches_meta[b'fine_label_names'][pred[0]].decode("utf-8")
Speaker.say_recognition(str_pred)