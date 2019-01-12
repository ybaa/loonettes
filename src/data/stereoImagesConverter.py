
import cv2
from matplotlib import pyplot as plt
from src.constants import DEPTH_MAPS_PATH
import datetime


class StereoImagesConverter:

    def __init__(self, frame_left, frame_right):
        self.frame_left = frame_left
        self.frame_right = frame_right

    def create_depth_map(self, save=False, show=True):
        # stereo_bm = cv2.StereoSGBM_create(numDisparities=16, blockSize=11)
        # stereo_bm = cv2.StereoBM_create(numDisparities=32, blockSize=15)
        # stereo_bm = cv2.StereoSGBM_create(minDisparity=-64,
        #                                   numDisparities=192,
        #                                   blockSize=5,
        #                                   P1=600,
        #                                   P2=2400,
        #                                   disp12MaxDiff=10,
        #                                   preFilterCap=4,
        #                                   uniquenessRatio=1,
        #                                   speckleRange=2,
        #                                   speckleWindowSize=150)
        stereo_bm = cv2.StereoSGBM_create(minDisparity=16,
                                          numDisparities=32,
                                          blockSize=15,
                                          P1=600,
                                          P2=2400,
                                          disp12MaxDiff=20,
                                          preFilterCap=16,
                                          uniquenessRatio=1,
                                          speckleRange=20,
                                          speckleWindowSize=100)
        conv_left, conv_right = self.convert_bgr_2_gray()
        depth_map = stereo_bm.compute(conv_left,conv_right)

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

    # ----------------------------------------------------------------------------------------------------
    def test_parameters(self):
        for i in range(16, 128, 16):
            for j in range(5, 31, 2):
                stereo_bm = cv2.StereoBM_create(numDisparities=i, blockSize=j)
                conv_left, conv_right = self.convert_bgr_2_gray()
                depth_map = stereo_bm.compute(conv_left, conv_right)
                print('current on numDisp=' + str(i) + ' and blockSize=' + str(j) + '\n')

                self.save_depth_map_with_params_in_name(path=DEPTH_MAPS_PATH, frame=depth_map, i=i, j=j)

    def save_depth_map_with_params_in_name(self, path, frame, i, j):
        ts = datetime.datetime.now().timestamp()
        ts = str(ts).replace(".", "")
        full_path = path + '/paramsTest/dm' + '_' + 'numDis' + str(i) + '_bs' + str(j) + '_' + ts + '.jpg'
        cv2.imwrite(full_path, frame)
        return
