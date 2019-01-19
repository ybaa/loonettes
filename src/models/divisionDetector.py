from matplotlib import pyplot as plt
import numpy as np
from src.models.cnnMyDataset import CNNMyDataset
from src.features.myDatasetHelper import MyDatasetHelper

class DivisionDetector:

    def __init__(self):
        self.divided_image = []
        pass

    def divide(self, img):
        height, width = img.shape[:2]
        new_height = int(height / 2)
        new_width = int(width / 2)

        up_left = img[:new_height, :new_width, ]
        up_right = img[:new_height, new_width:width, ]
        bottom_left = img[new_height:height, :new_width, ]
        bottom_right = img[new_height:height, new_width:width, ]

        # test printing
        # ch_0 = bottom_right[:,:,0]
        # ch_1 = bottom_right[:,:,1]
        # ch_2 = bottom_right[:,:,2]
        # merged = np.zeros((ch_0.shape[0], ch_0.shape[1], 3)).astype(int)
        # merged[:,:,0]=ch_0
        # merged[:,:,1]=ch_1
        # merged[:,:,2]=ch_2
        # plt.imshow(merged)
        # plt.show()

        self.divided_image.append(up_left)
        self.divided_image.append(up_right)
        self.divided_image.append(bottom_right)
        self.divided_image.append(bottom_left)

        # cnn_mine = CNNMyDataset()
        # predicted = cnn_mine.predict_single_image(up_left)

        return [up_left, up_right, bottom_left, bottom_right]

    def divide_recursively(self, depth, img):
        if depth == 0: return

        division = self.divide(img)
        for part in division:
            self.divide_recursively(depth - 1, part)

    def shift(self):
        print('shift')
