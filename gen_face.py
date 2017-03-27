from PIL import Image
import numpy as np
import fileinput
"""
读取图片
"""

class GenFace:
    def __init__(self):
        self.photo_count = 999
        self.name_count = 0
        self.name_mapping = {}
        for line in fileinput.input("name_mapping"):
            line = str(line).replace("\n","")
            self.name_mapping[line.split(" ")[0]] = line.split(" ")[1]
            self.name_count += 1
        self.i = 0
        self.j = 0

    def gen_captcha_text_and_image(self):
        # for i in range(self.photo_count):
        #     for j in range(self.name_count):
        #         captcha_image = Image.open("./me/" + str(i) + '.jpg')
        #         captcha_image = np.array(captcha_image)
        #         return j, captcha_image
        while True:
            try:
                # print("./me/" + self.name_mapping[self.intCoverToStr(self.j)] + '_' + str(self.i) + '.jpg')
                captcha_image = Image.open("./me/" + self.name_mapping[self.intCoverToStr(self.j)] + '_' + str(self.i) + '.jpg')
                captcha_image = captcha_image.resize((200, 200))
                aptcha_image = np.array(captcha_image)
                tmp = self.j
                self.j += 1
                if self.j % self.name_count == 0:
                    self.j = 0
                    self.i += 1
                return self.intCoverToStr(tmp), aptcha_image
            except Exception as err:
                print(err)
                self.j += 1
                if self.j % self.name_count == 0:
                    self.j = 0
                    self.i += 1
                continue
    def intCoverToStr(self, t):
        t = str(t)
        for i in range(4-len(t)):
            t = '0' + t
        return t

# gen_video = GenVideo()
# print(gen_video.intCoverToStr(10))




# gen_video = GenVideo()
# for i in range(100):
#     gen_video.gen_captcha_text_and_image()
