import tensorflow as tf
import numpy as np

def add_layer(input, input_size, output_size, activation_function=None):
    Weight = tf.Variable(tf.random_normal(input_size))
    biases = tf.Variable(tf.zeros(input_size) + 0.1)

    outputs = tf.add(tf.matmul(input, Weight), biases)
    if activation_function is None:
        return outputs
    else:
        return activation_function(outputs)

