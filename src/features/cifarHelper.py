import numpy as np


class CifarHelper:

    def __init__(self, train_batches, test_batch, labels_amount=10):
        self.i = 0

        self.all_train_batches = train_batches
        self.test_batch = [test_batch]

        self.training_images = None
        self.training_labels = None

        self.test_images = None
        self.test_labels = None

        self.labels_amount = labels_amount

    def set_up_images(self):
        print("Setting Up Training Images and Labels")

        self.training_images = np.vstack([d[b"data"] for d in self.all_train_batches])
        train_len = len(self.training_images)

        self.training_images = self.training_images.reshape(train_len, 3, 32, 32).transpose(0, 2, 3, 1) / 255
        self.training_labels = self.one_hot_encode(np.hstack([d[b"labels"] for d in self.all_train_batches]), self.labels_amount)

        print("Setting Up Test Images and Labels")

        self.test_images = np.vstack([d[b"data"] for d in self.test_batch])
        test_len = len(self.test_images)

        self.test_images = self.test_images.reshape(test_len, 3, 32, 32).transpose(0, 2, 3, 1) / 255
        self.test_labels = self.one_hot_encode(np.hstack([d[b"labels"] for d in self.test_batch]), self.labels_amount)

    def next_batch(self, batch_size):
        x = self.training_images[self.i:self.i + batch_size].reshape(batch_size, 32, 32, 3)
        y = self.training_labels[self.i:self.i + batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y

    def one_hot_encode(self,vec, vals=10):
        n = len(vec)
        out = np.zeros((n, vals))
        out[range(n), vec] = 1
        return out