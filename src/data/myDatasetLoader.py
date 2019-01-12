import pickle
import cv2
import glob
import csv
from src.constants import MY_DATASET_PATH


class MyDatasetLoader:

    def load_dataset(self):
        dirs = ['meta', 'train', 'test']

        all_data = [0, 1, 2]

        for i, direc in zip(all_data, dirs):
            all_data[i] = self.unpickle(MY_DATASET_PATH + direc)

        return [all_data[1]], all_data[2], all_data[0]

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dataset_dict = pickle.load(fo, encoding='bytes')
        return dataset_dict

    def pickle_data(self):
        images = [cv2.imread(file) for file in glob.glob(MY_DATASET_PATH + "*.jpg")]
        self.pickle(images, 'train')
        # self.pickle(images, 'test')

        # with open(MY_DATASET_PATH + 'batch_meta', newline='') as csvfile:
        #     reader = csv.reader(csvfile, delimiter=',')
        #     data = []
        #     for row in reader:
        #         data.append(row)
        #
        #     self.pickle(data[0], 'meta')

    def pickle(self, data, filename):
        with open(MY_DATASET_PATH + filename, 'wb') as fo:
            pickle.dump(data, fo, pickle.HIGHEST_PROTOCOL)
