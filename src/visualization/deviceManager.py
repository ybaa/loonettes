from src.data.camera import Camera
from src.data.stereoImagesConverter import StereoImagesConverter
from src.models.cnnMyDataset import CNNMyDataset
from src.models.divisionDetector import DivisionDetector
from src.features.myDatasetHelper import MyDatasetHelper
from collections import Counter
from src.statics.positionEnum import Positions
from src.visualization.speaker import Speaker
import tensorflow as tf
import numpy as np


class DeviceManager:

    def __init__(self):
        self.cam = Camera()
        self.cnn = CNNMyDataset()
        self.divider = DivisionDetector()
        self.speaker = Speaker()

    def run(self):

        with tf.Session() as sess:

            while True:
                single_img_divided_rescaled = self.__get_image()

                # predict
                single_img_predictions = self.__predict(single_img_divided_rescaled, sess)

                if len(single_img_predictions) > 0:

                    label, position = self.__get_class_and_position(single_img_predictions)

                    # say output
                    self.speaker.say_recognition(label, position)



    def __get_image(self):
        frame_l, frame_r = self.cam.capture_stereo()

        reshaped = MyDatasetHelper.resize_images([frame_l, frame_r], shape=(640, 480))

        # create and append with disparity map
        sic = StereoImagesConverter(reshaped[0], reshaped[1])
        disp_map = sic.create_disparity_map(show=False)
        appended_with_disp_map = sic.append_img_with_disparity_map(reshaped[0], disp_map)

        normalized = np.asanyarray(appended_with_disp_map) / 255

        # division
        self.divider.divided_image = []
        self.divider.divided_image.append(normalized)
        self.divider.divide_recursively(2, normalized)
        single_img_divided_rescaled = MyDatasetHelper.resize_images(self.divider.divided_image)

        return single_img_divided_rescaled

    def __predict(self, single_img_divided_rescaled, sess):
        single_img_predictions = []

        for index, img_part in enumerate(single_img_divided_rescaled):
            part_img_pred_val, prob_val = self.cnn.predict_single_image(img_part, sess)

            if part_img_pred_val is not None:
                single_img_predictions.append([part_img_pred_val[0], prob_val, index])

        return single_img_predictions

    def __get_class_and_position(self, single_img_predictions):
        classes = []
        for c in single_img_predictions:
            classes.append(c[0])

        # get class to return and its position
        most_common = Counter(classes).most_common(1)[0][0]

        positions = [p[2] for p in single_img_predictions if p[0] == most_common]

        # todo: find a better way of positioning
        positions_distribution = [0, 0, 0]
        for p in positions:
            if p in [0, 6, 8, 9, 11, 13, 15, 17, 19]:
                positions_distribution[Positions.MIDDLE.value] += 1
            elif p in [1, 3, 5, 7, 14, 16]:
                positions_distribution[Positions.LEFT.value] += 1
            else:
                positions_distribution[Positions.RIGHT.value] += 1

        final_position = positions_distribution.index(max(positions_distribution))

        return most_common, final_position