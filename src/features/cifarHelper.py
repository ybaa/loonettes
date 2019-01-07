import numpy as np


class CifarHelper:

    def __init__(self, train_batches, test_batch, batches_meta, labels_amount=10):
        self.i = 0

        self.all_train_batches = train_batches
        self.test_batch = [test_batch]

        self.training_images = None
        self.training_labels = None

        self.test_images = None
        self.test_labels = None

        self.labels_amount = labels_amount

        self.batches_meta = batches_meta

    def set_up_images(self):
        print("Setting Up Training Images and Labels")

        self.training_images = np.vstack([d[b"data"] for d in self.all_train_batches])
        train_len = len(self.training_images)

        self.training_images = self.training_images.reshape(train_len, 3, 32, 32).transpose(0, 2, 3, 1) / 255

        if self.labels_amount == 10:
            self.training_labels = self.one_hot_encode(np.hstack([d[b"labels"] for d in self.all_train_batches]))
        else:
            self.training_labels = self.one_hot_encode(np.hstack([d[b"fine_labels"] for d in self.all_train_batches]))

        print("Setting Up Test Images and Labels")

        self.test_images = np.vstack([d[b"data"] for d in self.test_batch])
        test_len = len(self.test_images)

        self.test_images = self.test_images.reshape(test_len, 3, 32, 32).transpose(0, 2, 3, 1) / 255

        if self.labels_amount == 10:
            self.test_labels = self.one_hot_encode(np.hstack([d[b"labels"] for d in self.test_batch]))
        else:
            self.test_labels = self.one_hot_encode(np.hstack([d[b"fine_labels"] for d in self.test_batch]))

    def next_batch(self, batch_size):
        x = self.training_images[self.i:self.i + batch_size].reshape(batch_size, 32, 32, 3)
        y = self.training_labels[self.i:self.i + batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y

    def one_hot_encode(self, vec):
        n = len(vec)
        out = np.zeros((n, self.labels_amount))
        out[range(n), vec] = 1
        return out