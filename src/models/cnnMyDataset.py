from src.models.cnnBase import CNNBase
import tensorflow as tf
# from src.models.divisionDetector import DivisionDetector
from src.data.myDatasetLoader import MyDatasetLoader
from src.features.myDatasetHelper import MyDatasetHelper
from tensorflow.python import debug as tf_debug
import numpy as np


class CNNMyDataset(CNNBase):

    def __init__(self, img_size=[120, 160], labels_amount=2, channels=3):
        pass
        self.x_shape = [None, img_size[0], img_size[1], channels]
        self.y_true_shape = [None, labels_amount]

        self.x = tf.placeholder(tf.float32, shape=self.x_shape)
        self.y_true = tf.placeholder(tf.float32, shape=self.y_true_shape)
        self.hold_prob = tf.placeholder(tf.float32)

        self.y_pred, self.train_step = self.create_layers(self.x, self.y_true, self.hold_prob)

    def run_learning_session(self, restore=False, save=False):

        my_dataset_helper = self.load_and_prepare_set()

        iter_number = 201  # it should be much bigger but this value is set for developing

        with tf.Session() as sess:

            if restore:
                self.restore_model(sess, '../models/myConvo/model.ckpt')
            else:
                sess.run(tf.global_variables_initializer())

            # sess = tf_debug.TensorBoardDebugWrapperSession(sess, "ybaa-pc:7000")

            for i in range(iter_number):
                batch = my_dataset_helper.next_batch(1)
                sess.run(self.train_step, feed_dict={self.x: batch[0],
                                                     self.y_true: batch[1],
                                                     self.hold_prob: 0.5})

                if i % 100 == 0:
                    print('Currently on step ' + str(i))
                    print('Accuracy is:')
                    # Test model
                    matches = tf.equal(tf.argmax(self.y_pred, 1), tf.argmax(self.y_true, 1))

                    acc = tf.reduce_mean(tf.cast(matches, tf.float32))

                    print(sess.run(acc, feed_dict={self.x: my_dataset_helper.test_images,
                                                   self.y_true: my_dataset_helper.test_labels,
                                                   self.hold_prob: 1.0}))
                    print('\n')

                if i == iter_number - 1 and save:
                    self.save_model(sess, '../models/myConvo/model.ckpt')


    def load_and_prepare_set(self, reshape_test_images=True):

        my_dataset_loader = MyDatasetLoader()
        # my_dataset_loader.pickle_data()
        training_batch, test_batch, batch_meta = my_dataset_loader.load_dataset()
        my_dataset_helper = MyDatasetHelper(training_batch, test_batch, batch_meta, labels_amount=2)
        my_dataset_helper.set_up_images(reshape_test_images=reshape_test_images)

        return my_dataset_helper


    def create_layers(self, x, y_true, hold_prob):
        convo_1 = self.convolutional_layer(x, shape=[4, 4, 3, 32])
        convo_1_pooling = self.max_pool_2by2(convo_1)

        convo_2 = self.convolutional_layer(convo_1_pooling, shape=[4, 4, 32, 64])
        convo_2_pooling = self.max_pool_2by2(convo_2)

        convo_2_flat = tf.reshape(convo_2_pooling, [-1, 30 * 40 * 64])

        full_layer_one = tf.nn.relu(self.normal_full_layer(convo_2_flat, 1024))

        full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)

        y_pred = self.normal_full_layer(full_one_dropout, 2)

        cross_entropy = self.create_loss_function(y_pred, y_true)

        train_step = self.create_optimizer(loss_function=cross_entropy)

        return y_pred, train_step

    def create_loss_function(self, y_pred, y_true):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred))
        return cross_entropy

    def create_optimizer(self, loss_function):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_step = optimizer.minimize(loss_function)
        return train_step

    def predict_single_image(self, img):
        with tf.Session() as sess:

            self.restore_model(sess, '../models/myConvo/model.ckpt')

            single_prediction = tf.argmax(self.y_pred, 1)
            img_reshaped = np.reshape(img, (1, 120, 160, 3))

            pred_val = sess.run(single_prediction, feed_dict={self.x: img_reshaped,
                                                              self.hold_prob: 1.0})
            return pred_val