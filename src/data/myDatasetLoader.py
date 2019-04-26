import pickle
import cv2
import glob
import csv
from src.features.myDatasetHelper import MyDatasetHelper
import itertools
from sklearn.model_selection import train_test_split

class MyDatasetLoader:

    def load_dataset_for_classification(self):
        dirs = ['all_images_p', 'labels_p']

        data = []
        meta = []

        for dir in dirs:
            data.append(self.unpickle('../data/raw/myDatasetClfs/' + dir))

        meta.append(self.unpickle('../data/raw/myDatasetClfs/batch_meta_p'))

        X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.2, random_state=42)

        return [X_train, y_train], [X_test, y_test], meta

    def load_dataset_for_detection(self):
        dirs_test = ['test', 'labels_p']

        test_data = []
        meta = []

        for dir in dirs_test:
            test_data.append(self.unpickle('../data/raw/myDatasetDetection/' + dir))

        meta.append(self.unpickle('../data/raw/myDatasetClfs/old/batch_meta_p'))

        return test_data, meta

    def pickle_data(self):
        base_path = '../data/raw/myDatasetClfs/'
        dirs = ['backpack', 'bike', 'book', 'chair', 'coach', 'cup', 'phone', 'skateboard']

        images = []
        classes = []

        for dir in dirs:
            path = base_path + dir + '/'

            filenames = glob.glob(path + "*.jpg")
            filenames.sort()

            single_class_images = [cv2.imread(file) for file in filenames]

            reshaped = MyDatasetHelper.resize_images(single_class_images, shape=(592, 410))

            dataset_appended_with_dm = MyDatasetHelper.crete_disparity_maps_serial(reshaped)

            images.append(dataset_appended_with_dm)
            single_class = [dir] * len(dataset_appended_with_dm)
            classes.append(single_class)

        classes = list(itertools.chain(*classes))
        images = list(itertools.chain(*images))

        self.pickle(images, 'all_images_p', base_path)
        self.pickle(classes, 'labels_p', base_path)
        self.pickle(dirs, 'batch_meta_p', base_path)


    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dataset_dict = pickle.load(fo, encoding='bytes')
        return dataset_dict

    def pickle(self, data, filename, path):
        with open(path + filename, 'wb') as fo:
            pickle.dump(data, fo, pickle.HIGHEST_PROTOCOL)
