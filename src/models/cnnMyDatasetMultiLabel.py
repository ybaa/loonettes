from src.models.cnnBase import CNNBase
import tensorflow as tf
# from src.models.divisionDetector import DivisionDetector
from src.data.myDatasetLoader import MyDatasetLoader
from src.features.myDatasetHelper import MyDatasetHelper
from tensorflow.python import debug as tf_debug
import numpy as np
from src.constants import LABELS_AMOUNT, CURR_CHANNELS, CURR_HEIGHT, CURR_WIDTH
import cv2
import time
import csv

class CNNMyDatasetMultiLabel(CNNBase):

    def __init__(self, curr_class='cokolwiek', img_size=[CURR_HEIGHT, CURR_WIDTH], channels=CURR_CHANNELS):
        pass
        self.curr_class = curr_class
        labels_amount = 2
        self.x_shape = [None, img_size[0], img_size[1], channels]
        self.y_true_shape = [None, labels_amount]

        self.x = tf.placeholder(tf.float32, shape=self.x_shape)
        self.y_true = tf.placeholder(tf.float32, shape=self.y_true_shape)
        self.hold_prob = tf.placeholder(tf.float32)

        self.y_pred, self.train_step = self.create_layers(self.x, self.y_true, self.hold_prob, channels)

    def run_learning_session(self, restore=False, save=False, save_csv=False):

        my_dataset_helper = self.load_and_prepare_set(reshape_test_images=False)

        iter_number = 41
        batch_size = 10

        data_to_csv = []

        with tf.Session() as sess:

            if restore:
                self.restore_model(sess, '../models/myConvo/' + str(CURR_CHANNELS) + 'chmlc/' + self.curr_class + 'model.ckpt')
            else:
                sess.run(tf.global_variables_initializer())

            # sess = tf_debug.TensorBoardDebugWrapperSession(sess, "ybaa-pc:7000")

            for i in range(iter_number):
                batch = my_dataset_helper.next_batch(batch_size)
                sess.run(self.train_step, feed_dict={self.x: batch[0],
                                                     self.y_true: batch[1],
                                                     self.hold_prob: 0.5})

                if i % 10 == 0:
                    print('Currently on step ' + str(i))
                    print('Accuracy is:')
                    # Test model

                    matches = tf.equal(tf.argmax(self.y_pred, 1), tf.argmax(self.y_true, 1))


                    acc = tf.reduce_mean(tf.cast(matches, tf.float32))
                    start = time.time()
                    acc_val = sess.run(acc, feed_dict={self.x: my_dataset_helper.test_images,
                                                   self.y_true: my_dataset_helper.test_labels,
                                                   self.hold_prob: 1.0})
                    end = time.time()
                    classification_time = (end - start) / len(my_dataset_helper.test_images)
                    data_to_csv.append([i, CURR_CHANNELS, 1, acc_val, classification_time])
                    print(acc_val)
                    print('\n')

                if i == iter_number - 1 and save:
                    self.save_model(sess, '../models/myConvo/' + str(CURR_CHANNELS) + 'chmlc/' + self.curr_class + '/model.ckpt')

            sess.close()

        if save_csv:
            with open('../reports/classification_eval_' + str(CURR_CHANNELS) + 'ch' + self.curr_class + 'MLC_b' + str(batch_size) + '.csv',
                      mode='w') as csv_file:
                writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for row in data_to_csv:
                    writer.writerow(row)

    def load_and_prepare_set(self, reshape_test_images=False, for_classification=True):

        my_dataset_loader = MyDatasetLoader()
        # classes = ['backpack', 'bike', 'book', 'chair', 'coach', 'cup', 'phone', 'skateboard']
        #
        # for cls in classes:
        #     my_dataset_loader.pickle_classification_data_for_mlc(cls)

        # my_dataset_loader.pickle_detection_data_for_mlc()

        if for_classification:
            training_batch, test_batch, batch_meta = my_dataset_loader.load_dataset_for_multi_label_classification(self.curr_class)
        else:
            test_batch, batch_meta = my_dataset_loader.load_dataset_for_multi_label_detection()
            training_batch = []

        my_dataset_helper = MyDatasetHelper(training_batch, test_batch, batch_meta, labels_amount=2)
        my_dataset_helper.set_up_images(reshape_test_images=reshape_test_images, detection=not for_classification)

        return my_dataset_helper

    def create_layers(self, x, y_true, hold_prob, channels):
        convo_1 = self.convolutional_layer(x, shape=[4, 4, channels, 32])
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

    def predict_single_image(self, img, sess):
        # with tf.Session() as sess:
            # sess = tf_debug.TensorBoardDebugWrapperSession(sess, "ybaa-pc:7000")

        classes = ['backpack', 'bike', 'book', 'chair', 'coach', 'cup', 'phone', 'skateboard']

        img_reshaped = np.reshape(img, (1, CURR_HEIGHT, CURR_WIDTH, CURR_CHANNELS))

        recognition = None
        recognition_prob = 0

        for (i, cls) in enumerate(classes):

            self.restore_model(sess, '../models/myConvo/' + str(CURR_CHANNELS) + 'chmlc/' + cls + '/model.ckpt')

            # cv2.imshow('a', img)
            # cv2.waitKey()
            # cv2.destroyAllWindows()


            # output from nn
            pred_one_hot = sess.run(self.y_pred, feed_dict={self.x: img_reshaped,
                                                            self.hold_prob: 1.0})

            # most probable one
            pred_max_index = tf.argmax(self.y_pred, 1)

            # index / class
            index = sess.run(pred_max_index, feed_dict={self.x: img_reshaped,
                                                        self.hold_prob: 1.0})

            # probability
            prob = sess.run(tf.nn.softmax(logits=pred_one_hot))

            pred_val_nn_output_value = pred_one_hot[0][index[0]]    # pure output from nn
            prob_val = prob[0][index[0]]

            # print(index)
            # print(pred_val_nn_output_value)
            # print(pred_one_hot[0])
            # print(prob)
            # print(prob_val)
            # print('-----------')

            if index == 1 and prob_val > 0.9:
                if prob_val > recognition_prob:
                    recognition = i

        return recognition

