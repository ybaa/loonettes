import pickle
import cv2
import glob
import csv
from src.features.myDatasetHelper import MyDatasetHelper
import itertools
from sklearn.model_selection import train_test_split
from src.constants import CURR_CHANNELS

class MyDatasetLoader:

    def load_dataset_for_classification(self):
        dirs = ['all_images_p', 'labels_p']

        data = []
        meta = []

        for dir in dirs:
            data.append(self.unpickle('../data/raw/myDatasetClfs/' + str(CURR_CHANNELS) + 'ch/' + dir))

        meta.append(self.unpickle('../data/raw/myDatasetClfs/' + str(CURR_CHANNELS) + 'ch/batch_meta_p'))

        X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.3, random_state=42)

        return [X_train, y_train], [X_test, y_test], meta

    def load_dataset_for_detection(self):
        dirs_test = ['images_p', 'labels_p']

        images = []
        meta = []

        for dir in dirs_test:
            images.append(self.unpickle('../data/raw/myDatasetDetection/' + str(CURR_CHANNELS) + 'ch/' + dir))

        meta.append(self.unpickle('../data/raw/myDatasetClfs/'+ str(CURR_CHANNELS) +'ch/batch_meta_p'))

        return images, meta

    def load_dataset_for_multi_label_classification(self, curr_class):
        dirs = ['_p', '_labels_p']

        data = []
        meta = []

        for dir in dirs:
            data.append(self.unpickle('../data/raw/myDatasetClfsMultiLabel/' + curr_class + dir))

        meta.append(self.unpickle('../data/raw/myDatasetClfsMultiLabel/batch_meta_p'))

        X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.3, random_state=42)

        return [X_train, y_train], [X_test, y_test], meta

    def pickle_classification_data(self):
        base_path = '../data/raw/myDatasetClfs/'
        dirs = ['backpack', 'bike', 'book', 'chair', 'coach', 'cup', 'phone', 'skateboard']

        images = []
        classes = []

        for dir in dirs:
            path = base_path + dir + '/'

            filenames = glob.glob(path + "*.jpg")
            filenames.sort()

            if CURR_CHANNELS == 3:
                single_class_images = [cv2.imread(file) for file in filenames if file[-5:] == 'L.jpg']
                reshaped = MyDatasetHelper.resize_images(single_class_images, shape=(160, 120))
                images.append(reshaped)
                single_class = [dir] * len(reshaped)
                classes.append(single_class)

            elif CURR_CHANNELS == 4:
                single_class_images = [cv2.imread(file) for file in filenames]
                reshaped = MyDatasetHelper.resize_images(single_class_images, shape=(160, 120))
                dataset_appended_with_dm = MyDatasetHelper.crete_disparity_maps_serial(reshaped)
                images.append(dataset_appended_with_dm)
                single_class = [dir] * len(dataset_appended_with_dm)
                classes.append(single_class)

            else:
                print('invalid channles')
                return None

        classes = list(itertools.chain(*classes))
        images = list(itertools.chain(*images))

        self.pickle(images, 'all_images_p', base_path)
        self.pickle(classes, 'labels_p', base_path)
        self.pickle(dirs, 'batch_meta_p', base_path)

    # moze do uwspólnienia z pickle_for_classification, ale to do przemyślenia
    def pickle_classification_data_for_mlc(self, curr_class):
        base_path = '../data/raw/myDatasetClfs'
        dirs = ['backpack', 'bike', 'book', 'chair', 'coach', 'cup', 'phone', 'skateboard']

        images = []
        classes = []

        for dir in dirs:
            path = base_path + '/' + dir + '/'

            filenames = glob.glob(path + "*.jpg")
            filenames.sort()

            if CURR_CHANNELS == 3:
                single_class_images = [cv2.imread(file) for file in filenames if file[-5:] == 'L.jpg']
                reshaped = MyDatasetHelper.resize_images(single_class_images, shape=(160, 120))
                images.append(reshaped)

                single_class = 1 * len(reshaped) if dir == curr_class else 0 * len(reshaped)
                classes.append(single_class)

            elif CURR_CHANNELS == 4:
                single_class_images = [cv2.imread(file) for file in filenames]
                reshaped = MyDatasetHelper.resize_images(single_class_images, shape=(160, 120))
                dataset_appended_with_dm = MyDatasetHelper.crete_disparity_maps_serial(reshaped)
                images.append(dataset_appended_with_dm)

                single_class = [True] * len(dataset_appended_with_dm) if dir == curr_class else [False] * len(dataset_appended_with_dm)
                classes.append(single_class)

            else:
                print('invalid channles')
                return None

        classes = list(itertools.chain(*classes))
        images = list(itertools.chain(*images))

        self.pickle(images, curr_class + '_p', base_path + 'MultiLabel/')
        self.pickle(classes, curr_class + '_labels_p', base_path + 'MultiLabel/')
        self.pickle([True, False], 'batch_meta_p', base_path + 'MultiLabel/')

    def pickle_detection_data(self):
        path = '../data/raw/myDatasetDetection/'

        filenames = glob.glob(path + "images/*.jpg")
        filenames.sort()

        if CURR_CHANNELS == 3:
            images = [cv2.imread(file) for file in filenames if file[-5:] == 'L.jpg']
            self.pickle(images, 'images_p', path)

        elif CURR_CHANNELS == 4:
            images = [cv2.imread(file) for file in filenames]
            reshaped = MyDatasetHelper.resize_images(images, shape=(640, 480))
            images_appended_with_dm = MyDatasetHelper.crete_disparity_maps_serial(reshaped)
            self.pickle(images_appended_with_dm, 'images_p', path)

        else:
            print('invalid channles')
            return None

        with open(path + "labels", newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            data = []
            for row in reader:
                data.append(row)

            self.pickle(data, 'labels_p', path)

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dataset_dict = pickle.load(fo, encoding='bytes')
        return dataset_dict

    def pickle(self, data, filename, path):
        with open(path + filename, 'wb') as fo:
            pickle.dump(data, fo, pickle.HIGHEST_PROTOCOL)
