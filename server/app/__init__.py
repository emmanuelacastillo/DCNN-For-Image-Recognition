# Author:
# Emmanuel A. Castillo
#
# Description:
# The service expects an image request.
# This image request will either contain an image
# that will initially filtered or left how it is.
# A CIFAR-10 based Deep Convolutional Neural Network
# model will process the image and classify it, returning
# the result to the client.

import cifar10_postprocess as dcnn
import pickle
import tensorflow as tf
import variableextraction as variables
import fullrun

from flask import request, jsonify, Flask

# Initialize flask server
app = Flask(__name__)

def create_app():

    # Endpoint that expects a partially filtered image
    @app.route('/tensorflow/partial', methods=['POST'])
    def partial_processing():
        # Get post request and extract preprocessed image numpy array
        data = dict(request.files).get('file')[0]
        print('Request: ')
        print(data)
        preprocessed_image = pickle.load(data)

        # Initialize graph with preprocessed image
        post_processed_image = dcnn.setPreprocessedImage(preprocessed_image)
        variables.getVariables()
        init_op = tf.global_variables_initializer()

        # Run remainder of cifar-10 neural network
        with tf.Session() as sess:
            sess.run(init_op)
            logits = sess.run(post_processed_image)

        # Calculate predictions and return results
        images, labels = dcnn.inputs(eval_data=True)
        prediction = tf.nn.in_top_k(logits, labels, 1)
        print(str(prediction[0]))
        response = jsonify({'classification': 'success'})
        response.status_code = 201
        return response

    # Endpoint that expects an unfiltered image
    @app.route('/tensorflow/full', methods=['POST'])
    def full_processing():
        # Get post request and extract preprocessed image numpy array
        data = dict(request.files).get('file')[0]
        print('Request: ')
        print(data)
        image_to_process = pickle.load(data)

        # Calculate predictions and return results
        prediction = fullrun.send_processed_image(image=image_to_process)
        response = jsonify({'classification': 'success'})
        response.status_code = 201
        return response

    return app