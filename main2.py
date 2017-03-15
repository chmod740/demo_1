import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from captcha.image import ImageCaptcha
import random

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

"""创建一个显示图片的容器，并且显示图片"""
f = plt.figure()
ax = f.add_subplot(111)
ax.text(0.1, 0.9, text, ha="center", va="center", transform=ax.transAxes)
plt.imshow(image)
plt.show()

