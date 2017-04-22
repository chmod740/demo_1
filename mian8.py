
import tensorflow as tf
import numpy as np
from gen_face import GenFace

def add_layer(input, input_size, output_size, activation_function=None):
    # 矩阵
    Weight = tf.Variable(tf.random_normal([input_size, output_size]))
    biases = tf.Variable(tf.zeros([1, output_size]))
    output = tf.add(tf.matmul(input, Weight), biases)
    if activation_function is None:
        return output
    else:
        return activation_function(output)

def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img

gen_face = GenFace()
def get_x_y_data():
    text, image = gen_face.gen_captcha_text_and_image()
    image = convert2gray(image)
    image = image.reshape(-1, 1600)/255
    return image, text2vec(text)

def text2vec(text):
    vector = np.zeros(3)
    for i in range(3):
        vector[i] = int(text[i])
    return vector

def vec2text(vec):
    text = ''
    for i in range(3):
        text += str(vec[i])
    return text


xs = tf.placeholder(tf.float32, [None, 1600])
ys = tf.placeholder(tf.float32, [3])
keep_prob = tf.placeholder(tf.float32)

layer_1 = add_layer(xs, 1600, 1000, activation_function=tf.nn.relu)
# layer_1 = tf.nn.dropout(layer_1, keep_prob)

layer_2 = add_layer(layer_1, 1000, 1000, activation_function=tf.nn.relu)
# layer_2 = tf.nn.dropout(layer_2, keep_prob)

layer_3 = add_layer(layer_2, 1000, 1000, activation_function=tf.nn.relu)
# layer_3 = tf.nn.dropout(layer_3, keep_prob)

prediction = add_layer(layer_1, 1000, 3, activation_function=tf.nn.softmax)
prediction = tf.reshape(prediction, [3])
# prediction = tf.nn.dropout(prediction, keep_prob)

loss = tf.reduce_mean(tf.square(ys-prediction))
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, ys))
# train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
train_step = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(loss)
init = tf.global_variables_initializer()


#训练
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for i in range(2000):
        x_data, y_data = get_x_y_data()
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data, keep_prob: 1.})
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data, keep_prob: 1.}))
    saver.save(sess, "./crack_capcha.model", global_step=i)
    print("end")
# # 验证
# def crack():
#     saver = tf.train.Saver()
#     with tf.Session() as sess:
#         for i in  range(100):
#             saver.restore(sess, tf.train.latest_checkpoint('.'))
#             x_data, y_data = get_x_y_data()
#             # print(y_data)
#             output = sess.run(prediction, feed_dict={xs: x_data})
#             # print("输出" + str(output))
#             print("loss" + str(sess.run(loss, feed_dict={xs: x_data, ys: y_data})))
#             # print("YS" + str(y_data))
#             print()
#
#
#
#
# crack()
#
#