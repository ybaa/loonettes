import cv2
from matplotlib import pyplot as plt
import datetime
import numpy as np


class StereoImagesConverter:

    def __init__(self, frame_left, frame_right):
        self.frame_left = frame_left
        self.frame_right = frame_right
        self.disparity_map = None

    def create_disparity_map(self, save=False, show=True):

        # left_matcher = cv2.StereoSGBM_create(minDisparity=0,
        #                                   numDisparities=32,
        #                                   blockSize=11,
        #                                   P1=600,
        #                                   P2=2400,
        #                                   disp12MaxDiff=20,
        #                                   preFilterCap=16,
        #                                   uniquenessRatio=1,
        #                                   speckleRange=20,
        #                                   speckleWindowSize=100)



        window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

        left_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16,  # max_disp has to be dividable by 16
            blockSize=5,
            P1=8 * 3 * window_size ** 2,
            P2=32 * 3 * window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

        # FILTER Parameters
        lmbda = 80000
        sigma = 1.2

        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)

        #wls filter
        disp_l = left_matcher.compute(self.frame_left, self.frame_right)
        disp_r = right_matcher.compute(self.frame_right, self.frame_left)
        disp_l = np.int16(disp_l)
        disp_r = np.int16(disp_r)
        filtered_img = wls_filter.filter(disp_l, self.frame_left, None, disp_r)  # important to put "imgL" here!!!

        filtered_img = cv2.normalize(src=filtered_img, dst=filtered_img, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        filtered_img = np.uint8(filtered_img)

        self.disparity_map = filtered_img

        if show:
            self.show_disparity_map(filtered_img)

        if save:
            self.save_disparity_map(path='../data/interim/disparity_maps', frame=filtered_img)
            print('disparity map saved')

        return filtered_img

    # not working, still todo
    def calculate_distance(self, dm):
        fl = 580
        offset = 0.062

        # from the middle
        w = int(dm.shape[0] / 2)
        h = int(dm.shape[1] / 2)
        v = dm[w][h]

        distance = offset * fl / v
        print(distance)

        # d = 0.2
        # f = d * offset / v
        # print(f)

        return distance


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
        # plt.imshow(disparity_map, 'gray')
        # plt.show()
        cv2.imshow('Disparity Map', disparity_map)
        cv2.waitKey()
        cv2.destroyAllWindows()

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

    # ----------------------------------------------------------------------------------------------------
    def test_parameters(self):
        for i in range(16, 128, 16):
            for j in range(5, 31, 2):
                stereo_bm = cv2.StereoSGBM_create(numDisparities=i, blockSize=j)
                conv_left, conv_right = self.convert_bgr_2_gray()
                depth_map = stereo_bm.compute(conv_left, conv_right)
                print('current on numDisp=' + str(i) + ' and blockSize=' + str(j) + '\n')

                self.save_depth_map_with_params_in_name(path='../data/interim/disparity_maps', frame=depth_map, i=i, j=j)

    def save_depth_map_with_params_in_name(self, path, frame, i, j):
        ts = datetime.datetime.now().timestamp()
        ts = str(ts).replace(".", "")
        full_path = path + '/paramsTest/dm' + '_' + 'numDis' + str(i) + '_bs' + str(j) + '_' + ts + '.jpg'
        cv2.imwrite(full_path, frame)
        return


