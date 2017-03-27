import tensorflow as tf
import numpy as np
from gen_face import GenFace

def add_layer(input, input_size, output_size, activation_function=None):
    # 矩阵
    Weight = tf.Variable(tf.random_normal([input_size, output_size]))
    biases = tf.Variable(tf.zeros([1,output_size]))
    output = tf.add(tf.matmul(Weight,input_size), biases)
    if activation_function is None:
        return output
    else:
        return activation_function(output)

gen_face = GenFace()
text, image = gen_face.gen_captcha_text_and_image()

xs = tf.placeholder(tf.int32, [1, ])


layer_1 = add_layer()