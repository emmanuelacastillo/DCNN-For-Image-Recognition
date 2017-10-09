# Author:
# Emmanuel A. Castillo
#
# Description: This client will handle sending an image
# to an edge device that will classify the image.
# The sent image will either be filtered initially
# by being processed through part of the CIFAR-10
# Deep Convolutional Neural Network (DCNN) model or sent unfiltered.

import cifar10_preprocess
import numpy as np
import pickle
import requests
import tensorflow as tf
import time as time
import variableextraction as variables

from PIL import Image

# File paths and url constants
image_path = 'C:\\Users\\emman\\Desktop\\tensorflow\\TensorFlowClient\\images\\test8.jpg'
pre_process_image_path = 'C:\\Users\\emman\\Desktop\\tensorflow\\TensorFlowClient\\images\\preprocessimage.pkl'
image_partial_url = 'http://127.0.0.1:5000/tensorflow/partial'
image_full_url = 'http://127.0.0.1:5000/tensorflow/full'

# Function that will be called if the image will be filtered by part of the
# CIFAR-10 DCNN model prior to be sent to the edge device.
def send_for_remainder_processed_image():
    # Extract and expand image for dcnn
    img = Image.open(image_path)
    image_data = np.asarray(img, dtype='float32')
    expand_img = tf.expand_dims(image_data, 0)

    # Initialize graph with expanded image and extract weights and bias from .meta file
    preprocessed_image = cifar10_preprocess.setImageToGraph(expand_img)
    variables.getVariables()
    init_op = tf.global_variables_initializer()

    start_time = time.time()

    # Run partial cifar-10 neural network
    with tf.Session() as sess:
        sess.run(init_op)
        np_array_image = sess.run(preprocessed_image)

    print('Size of processed image: ' + str(np_array_image.size))

    # Send partial processed image to edge device for further processing
    pickle.dump(np_array_image, open(pre_process_image_path, 'wb'))
    resp = requests.post(image_partial_url, files={'file': open(pre_process_image_path, 'rb')})
    print(resp.content)

    duration = time.time() - start_time
    print('Compute Time: ' + str(duration))

# Function that will be called if the image without any pre-processing to the edge device.
def send_for_full_processed_image():
    # Extract and expand image for dcnn
    img = Image.open(image_path)
    image_data = np.asarray(img, dtype='float32')

    print('Size of processed image: ' + str(image_data.size))

    start_time = time.time()

    # Send full image to edge device for further processing
    pickle.dump(image_data, open(pre_process_image_path, 'wb'))
    resp = requests.post(image_full_url, files={'file': open(pre_process_image_path, 'rb')})
    print(resp.content)

    duration = time.time() - start_time
    print('Compute Time: ' + str(duration))

# Call function that will be used based on the test scenario.
send_for_full_processed_image()