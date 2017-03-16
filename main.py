import tensorflow as tf
import numpy as np

# 导入或者随机定义训练的数据 x 和 y：
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
y = Weights*x_data + biases
loss = tf.reduce_mean(tf.square(y-y_data))
# 选择一个基本的optimizer
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 使loss达到最小
train = optimizer.minimize(loss)
# 初始化所有变量
init = tf.initialize_all_variables()
# 将session指针指向需要处理的地方
sess = tf.Session()
# 开始运行
sess.run(init)
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))

"""添加神经层"""

def add_layer(inputs, in_size, out_size,  activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


