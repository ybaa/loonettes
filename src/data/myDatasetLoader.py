import pickle
import cv2
import glob
import csv


class MyDatasetLoader:

    def load_dataset(self):
        dirs_train = ['train', 'labels_p']
        dirs_test = ['test', 'labels_p']

        train_data = []
        test_data = []
        meta = []

        for dir in dirs_train:
            train_data.append(self.unpickle('../data/raw/myDatasetClfs/train/' + dir))

        for dir in dirs_test:
            test_data.append(self.unpickle('../data/raw/myDatasetClfs/test/' + dir))

        meta.append(self.unpickle('../data/raw/myDatasetClfs/batch_meta_p'))

        return train_data, test_data, meta

    def pickle_data(self):
        path = '../data/raw/myDatasetClfs/train/'
        images = [cv2.imread(file) for file in glob.glob(path + "*.jpg")]
        self.pickle(images, 'train', path)

        with open(path + 'labels' , newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            data = []
            for row in reader:
                data.append(row)

            self.pickle(data[0], 'labels_p', path)

        # pickle test
        path = '../data/raw/myDatasetClfs/test/'
        images = [cv2.imread(file) for file in glob.glob(path + "*.jpg")]
        self.pickle(images, 'test', path)

        with open(path + 'labels' , newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            data = []
            for row in reader:
                data.append(row)

            self.pickle(data[0], 'labels_p', path)

        # pickle meta
        path = '../data/raw/myDatasetClfs/batch_meta'
        with open(path , newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            data = []
            for row in reader:
                data.append(row)

            self.pickle(data[0], '_p', path)


    # def load_dataset(self):
    #     dirs = ['meta', 'train', 'test']
    #
    #     all_data = [0, 1, 2]
    #
    #     for i, direc in zip(all_data, dirs):
    #         all_data[i] = self.unpickle(MY_DATASET_DETECTION_PATH + direc)
    #
    #     return all_data[1], all_data[2], all_data[0]

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dataset_dict = pickle.load(fo, encoding='bytes')
        return dataset_dict

    # def pickle_data(self):
    #     images = [cv2.imread(file) for file in glob.glob(MY_DATASET_DETECTION_PATH + "*.jpg")]
    #     self.pickle(images, 'train')
    #
    #     # pickle test
    #     # self.pickle(images, 'test')
    #
    #     # pickle meta
    #     # with open(MY_DATASET_PATH + 'batch_meta', newline='') as csvfile:
    #     #     reader = csv.reader(csvfile, delimiter=',')
    #     #     data = []
    #     #     for row in reader:
    #     #         data.append(row)
    #     #
        #     self.pickle(data[0], 'meta')

    def pickle(self, data, filename, path):
        with open(path + filename, 'wb') as fo:
            pickle.dump(data, fo, pickle.HIGHEST_PROTOCOL)
