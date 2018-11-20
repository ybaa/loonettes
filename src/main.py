from src.data.camera import Camera
from src.constants import STEREO_CAM, STEREO_CAPTURES_FROM_CAM_PATH
from src.data.stereoImagesConverter import StereoImagesConverter
from src.models.cnnCifar import CNNCifar


cnn = CNNCifar()
cnn.run_session()

# cam = Camera(STEREO_CAM['LEFT'], STEREO_CAM['RIGHT'])
# frame_left, frame_right = cam.capture_stereo(STEREO_CAPTURES_FROM_CAM_PATH, save=False)

# stereo_img = StereoImagesConverter(frame_left, frame_right)
# stereo_img.create_depth_map()