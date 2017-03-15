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


text, image = gen_captcha_text_and_image()


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



