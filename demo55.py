import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
learning_rate = 0.01
training_epochs = 100

x_train = np.linspace(-1, 1, 101)
print(x_train)
y_train = 2 * x_train + np.random.randn(*x_train.shape) * 0.33
print(y_train)


def model(X, w):
    return tf.multiply(X, w)


X = tf.compat.v1.placeholder(tf.float32)
Y = tf.compat.v1.placeholder(tf.float32)

w = tf.Variable(0.0, name="weights")
y_model = model(X, w)

plt.scatter(x_train, y_train)
plt.show()