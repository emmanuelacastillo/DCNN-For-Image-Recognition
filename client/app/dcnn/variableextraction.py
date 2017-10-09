# Author:
# Emmanuel A. Castillo
#
# Description:
# TensorFlow provides the ability to save and reuse
# Deep Convolutional Neural Network models and weights.
# This script provides the ability to extract those
# information.

import tensorflow as tf


def getVariables():
    with tf.Session() as sess:
      new_saver = tf.train.import_meta_graph('C:/tmp/cifar10_train/model.ckpt-102827.meta')
      new_saver.restore(sess, 'C:/tmp/cifar10_train/model.ckpt-102827')