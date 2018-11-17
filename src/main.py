from src.data.camera import Camera
from src.constants import STEREO_CAM, STEREO_CAPTURES_FROM_CAM_PATH

cam = Camera(STEREO_CAM['LEFT'], STEREO_CAM['RIGHT'])
cam.capture_stereo(STEREO_CAPTURES_FROM_CAM_PATH, save=True)
