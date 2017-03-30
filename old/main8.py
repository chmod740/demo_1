import numpy as np

def text2vec(text):
    vector = np.zeros(40)
    for i in range(4):
        vector[i*10+int(text[i])] = 1
    return vector

def vec2text(vec):
    text = ''
    for i in range(40):
        if vec[i] == 1:
            text += str((i) % 10)
    return text
print(text2vec('0000'))
print(vec2text(text2vec('4561')))