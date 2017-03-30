# -*- coding:utf-8 -*-
import tensorflow as tf
import dataset


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def main():
    with tf.Session() as sess:
        data = dataset.read_data_sets('d:\\python code\\snh\\data\\ocr_test.pkl')
        x = tf.placeholder(tf.float32, shape=[None, 1800])
        y_ = tf.placeholder(tf.float32, shape=[None, 40])
        x_image = tf.reshape(x, [-1, 60, 30, 1])

        # 卷积层1
        w_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        # 卷积层2
        w_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        # 全连接层1
        w_fc1 = weight_variable([15 * 8 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 15 * 8 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        w_fc2 = weight_variable([1024, 40])
        b_fc2 = bias_variable([40])
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

        cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
        train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(tf.reshape(y_conv, [-1, 4, 10]), 2),
                                      tf.argmax(tf.reshape(y_, [-1, 4, 10]), 2))
        # correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            batch = data.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


if __name__ == '__main__':
    main()
