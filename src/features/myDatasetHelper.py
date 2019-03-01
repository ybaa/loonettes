from src.data.stereoImagesConverter import StereoImagesConverter
import cv2
import numpy as np
from sklearn import preprocessing


class MyDatasetHelper:
    def __init__(self, train_batch, test_batch, batches_meta, labels_amount=2):
        self.i = 0

        self.test_batch = [test_batch]

        self.training_images = train_batch[0] if len(train_batch) > 0 else []
        self.training_labels = train_batch[1] if len(train_batch) > 0 else []

        self.test_images = test_batch[0]
        self.test_labels = test_batch[1]

        self.labels_amount = labels_amount

        self.batches_meta = batches_meta
        self.training_batches_encoded = None
        self.test_batches_encoded = None

        self.le = preprocessing.LabelEncoder()

    def set_up_images(self, reshape_test_images=True):
        if len(self.training_images) > 0:
            print("Setting Up Training Images and Labels")
            self.training_images = self.resize_images(self.training_images)

            # to numpy array and normalize
            self.training_images = np.asanyarray(self.training_images) / 255

            # encode labels
            self.le.fit(self.batches_meta[0])
            self.training_batches_encoded = self.le.transform(self.training_labels)

            self.training_labels = self.one_hot_encode(np.asanyarray(self.training_batches_encoded))

        if len(self.test_images) > 0:
            print("Setting Up Test Images and Labels")
            if reshape_test_images:
                self.test_images = self.resize_images(self.test_images)

            # to numpy array and normalize
            self.test_images = np.asanyarray(self.test_images) / 255

            # encode labels
            self.le.fit(self.batches_meta[0])
            self.test_batches_encoded = self.le.transform(self.test_labels)

            self.test_labels = self.one_hot_encode(np.asanyarray(self.test_batches_encoded))

    def next_batch(self, batch_size):
        x = self.training_images[self.i:self.i + batch_size].reshape(batch_size, 120, 160, 3)
        y = self.training_labels[self.i:self.i + batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y

    def one_hot_encode(self, vec):
        n = len(vec)
        out = np.zeros((n, self.labels_amount))
        out[range(n), vec] = 1
        return out

    @staticmethod
    def resize_images(images, shape=(160, 120)):

        rescaled = []

        for img in images:
            rescaled.append(cv2.resize(img, shape))

        return rescaled

    @staticmethod
    def crete_disparity_maps_serial(images):

        dataset_appended_with_disparity_maps = []

        for i in range(0, len(images) - 1, 2):
            stereo_img = StereoImagesConverter(images[i], images[i + 1])
            disparity_map = stereo_img.create_disparity_map()

            frame_left_appended = stereo_img.append_img_with_disparity_map(images[i], disparity_map)
            frame_right_appended = stereo_img.append_img_with_disparity_map(images[i + 1], disparity_map)

            dataset_appended_with_disparity_maps.append(frame_left_appended)
            dataset_appended_with_disparity_maps.append(frame_right_appended)

        return dataset_appended_with_disparity_maps
