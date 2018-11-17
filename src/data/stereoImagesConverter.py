import cv2
from matplotlib import pyplot as plt
from src.constants import DEPTH_MAPS_PATH
import datetime


class StereoImagesConverter:

    def __init__(self, frame_left, frame_right):
        self.frame_left = frame_left
        self.frame_right = frame_right

    def create_depth_map(self, save=False, show=True):
        # stereoBM = cv2.StereoSGBM_create(numDisparities=16, blockSize=11)
        stereo_bm = cv2.StereoBM_create(numDisparities=16, blockSize=9)
        conv_left, conv_right = self.convert_bgr_2_gray()
        depth_map= stereo_bm.compute(conv_left, conv_right)

        if show:
            self.show_depth_map(depth_map)

        if save:
            self.save_depth_map(path=DEPTH_MAPS_PATH, frame=depth_map)

        return depth_map

    def convert_bgr_2_gray(self):
        img_left = cv2.cvtColor(self.frame_left, cv2.COLOR_BGR2GRAY)
        img_right = cv2.cvtColor(self.frame_right, cv2.COLOR_BGR2GRAY)
        return img_left, img_right

    def save_depth_map(self, path, frame):
        ts = datetime.datetime.now().timestamp()
        ts = str(ts).replace(".", "")
        full_path = path + '/dm' + '_' + ts + '.jpg'
        cv2.imwrite(full_path, frame)
        return

    def show_depth_map(self, depth_map):
        plt.imshow(depth_map, 'gray')
        plt.show()