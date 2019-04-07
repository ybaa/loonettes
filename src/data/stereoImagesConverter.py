import cv2
from matplotlib import pyplot as plt
import datetime
import numpy as np


class StereoImagesConverter:

    def __init__(self, frame_left, frame_right):
        self.frame_left = frame_left
        self.frame_right = frame_right

    def create_disparity_map(self, save=False, show=True):
        # stereoBM = cv2.StereoSGBM_create(numDisparities=16, blockSize=11)
        stereo_bm = cv2.StereoBM_create(numDisparities=16, blockSize=9)
        conv_left, conv_right = self.convert_bgr_2_gray()
        depth_map = stereo_bm.compute(conv_left, conv_right)

        if show:
            self.show_disparity_map(depth_map)

        if save:
            self.save_disparity_map(path='../data/interim/disparity_maps', frame=depth_map)
            print('disparity map saved')

        return depth_map

    def convert_bgr_2_gray(self):
        img_left = cv2.cvtColor(self.frame_left, cv2.COLOR_BGR2GRAY)
        img_right = cv2.cvtColor(self.frame_right, cv2.COLOR_BGR2GRAY)
        return img_left, img_right

    def save_disparity_map(self, path, frame):
        ts = datetime.datetime.now().timestamp()
        ts = str(ts).replace(".", "")
        full_path = path + '/dm' + '_' + ts + '.jpg'
        cv2.imwrite(full_path, frame)
        return

    def show_disparity_map(self, disparity_map):
        plt.imshow(disparity_map, 'gray')
        plt.show()

    def append_img_with_disparity_map(self, img, disparity_map):
        channel_0 = img[:, :, 0]
        channel_1 = img[:, :, 1]
        channel_2 = img[:, :, 2]

        merged = np.zeros((img.shape[0], img.shape[1], 4)).astype(int)
        merged[:, :, 0] = channel_0
        merged[:, :, 1] = channel_1
        merged[:, :, 2] = channel_2
        merged[:, :, 3] = disparity_map

        return merged

    def rectify_image(self, img):
        cv2.imshow('image', img)
        cv2.waitKey(0)

