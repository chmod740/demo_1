import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from captcha.image import ImageCaptcha
import random
import tensorflow as tf

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def get_random_captcha_text(keyset = number, text_size = 4):
    """
    得到随机串
    :param keyset:
    :param text_size:
    :return:
    """
    captcha = []
    for i in range(text_size):
        c = random.choice(number)
        captcha.append(c)
    return captcha

def gen_captcha_text_and_image():
    """
    生成验证码图片以及文字
    :return:
    """
    # 生成验证码的类
    image = ImageCaptcha()
    captcha_text = get_random_captcha_text()

    captcha_text = ''.join(captcha_text)
    captcha = image.generate(captcha_text)
    """
    生成图片的类
    """
    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)

    return captcha_text, captcha_image





# """创建一个显示图片的容器，并且显示图片"""
# f = plt.figure()
# ax = f.add_subplot(111)
# ax.text(0.1, 0.9, text, ha="center", va="center", transform=ax.transAxes)
# plt.imshow(image)
# plt.show()


def convert2gray(img):
    if len(img.shape) > 2:
        """
        [60,160,3] => [60, 160]
        """
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img

def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    # 权重
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # 偏置单元
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # Σ 权重*+偏置单元 => 激励函数
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        # 使用激励函数来计算结果
        outputs = activation_function(Wx_plus_b)
    return outputs


text, image = gen_captcha_text_and_image()

"""
将图像转换成灰度图像
"""
image = convert2gray(image)

x = tf.reshape(image, shape=[-1, 1])


def text2vecot(text):
    vector = np.zeros(40)
    for i in range(4):

# 1.训练的数据
# Make up some real data

#
# y_data = np.square(x_data) - 0.5 + noise
#
# # 2.定义节点准备接收数据
# # define placeholder for inputs to network
# xs = tf.placeholder(tf.float32, [None, 1])
# ys = tf.placeholder(tf.float32, [None, 1])
#
# # 3.定义神经层：隐藏层和预测层
# # add hidden layer 输入值是 xs，在隐藏层有 10 个神经元
# l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# # add output layer 输入值是隐藏层 l1，在预测层输出 1 个结果
# prediction = add_layer(l1, 10, 1, activation_function=None)
#
# # 4.定义 loss 表达式
# # the error between prediciton and real data
# loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
#                      reduction_indices=[1]))
# # 5.选择 optimizer 使 loss 达到最小
# # 这一行定义了用什么方式去减少 loss，学习率是 0.1
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#
# # important step 对所有变量进行初始化
# init = tf.initialize_all_variables()
# sess = tf.Session()
# # 上面定义的都没有运算，直到 sess.run 才会开始运算
# sess.run(init)
#
# # 迭代 1000 次学习，sess.run optimizer
# for i in range(1000):
#     # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
#     sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
#     if i % 50 == 0:
#         # to see the step improvement
#         print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
