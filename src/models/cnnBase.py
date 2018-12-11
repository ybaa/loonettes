import tensorflow as tf


class CNNBase:

    def init_weights(self, shape, stddev=0.1):
        init_random_dist = tf.truncated_normal(shape, stddev=stddev)
        return tf.Variable(init_random_dist)

    def init_bias(self, shape):
        init_bias_vals = tf.constant(0.1, shape=shape)
        return tf.Variable(init_bias_vals)

    def conv2d(self, x, W):
        # x - input tensor [batch, height, width, channels]
        # W - kernel (filter) [height, width, channels in, channels out]
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2by2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def convolutional_layer(self, input_x, shape):
        W = self.init_weights(shape)
        b = self.init_bias([shape[3]])
        return tf.nn.relu(self.conv2d(input_x, W) + b)

    def normal_full_layer(self, input_layer, size):
        input_size = int(input_layer.get_shape()[1])
        W = self.init_weights([input_size, size])
        b = self.init_bias([size])
        return tf.matmul(input_layer, W) + b

    def save_graph(self):
        writer = tf.summary.FileWriter('../reports')
        writer.add_graph(tf.get_default_graph())

    def save_model(self, session, dir):
        saver = tf.train.Saver()
        path = saver.save(session, dir)
        print('model saved in: ' + str(path))

    def restore_model(self, session, dir):
        saver = tf.train.Saver()
        saver.restore(session, dir)
