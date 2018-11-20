import tensorflow as tf
from src.models.cnnBase import CNNBase
from src.data.cifarLoader import CifarLoader
from src.features.cifarHelper import CifarHelper


class CNNCifar(CNNBase):

    def __init__(self, img_size=[32,32], labels_amount=10, channels=3):
        pass
        self.x_shape = [None, img_size[0], img_size[1], channels]
        self.y_true_shape = [None, labels_amount]

    def run_session(self):

        x = tf.placeholder(tf.float32, shape=self.x_shape)
        y_true = tf.placeholder(tf.float32, shape=self.y_true_shape)
        hold_prob = tf.placeholder(tf.float32)

        cifar_helper = self.load_and_prepare_set()

        y_pred, train = self.create_layers(x, y_true, hold_prob)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(300):
                batch = cifar_helper.next_batch(50)
                sess.run(train, feed_dict={x: batch[0],
                                           y_true: batch[1],
                                           hold_prob: 0.5})

                # PRINT OUT A MESSAGE EVERY 100 STEPS
                if i % 50 == 0:
                    print('Currently on step {}'.format(i))
                    print('Accuracy is:')
                    # Test the Train Model
                    matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))

                    acc = tf.reduce_mean(tf.cast(matches, tf.float32))

                    print(sess.run(acc, feed_dict={x: cifar_helper.test_images,
                                                   y_true: cifar_helper.test_labels,
                                                   hold_prob: 1.0}))
                    print('\n')

    def load_and_prepare_set(self):
        loader = CifarLoader()
        training_batches, test_batch, batch_meta = loader.load_data()

        cifar_helper = CifarHelper(training_batches, test_batch)
        cifar_helper.set_up_images()
        return cifar_helper

    def create_layers(self, x, y_true, hold_prob):
        convo_1 = self.convolutional_layer(x, shape=[4, 4, 3, 32])
        convo_1_pooling = self.max_pool_2by2(convo_1)

        convo_2 = self.convolutional_layer(convo_1_pooling, shape=[4, 4, 32, 64])
        convo_2_pooling = self.max_pool_2by2(convo_2)

        convo_2_flat = tf.reshape(convo_2_pooling, [-1, 8 * 8 * 64])

        full_layer_one = tf.nn.relu(self.normal_full_layer(convo_2_flat, 1024))

        full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)

        y_pred = self.normal_full_layer(full_one_dropout, 10)

        cross_entropy = self.create_loss_function(y_pred, y_true)

        train_optimizer = self.create_optimizer(loss_function=cross_entropy)

        return y_pred, train_optimizer

    def create_loss_function(self, y_pred, y_true):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred))
        return cross_entropy

    def create_optimizer(self, loss_function):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train = optimizer.minimize(loss_function)
        return train

