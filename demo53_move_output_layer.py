import numpy
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

dataset1 = numpy.loadtxt("data/diabetes.csv", skiprows=1, delimiter=",")
print(dataset1.shape)

inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]
print(inputList.shape)
print(resultList.shape)

model = Sequential()
model.add(Dense(14, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
output_layer = Dense(1, activation='sigmoid')
model.add(output_layer)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
print("before training, model coef:", output_layer.get_weights()[0])
print("before training, model intercept:", output_layer.get_weights()[1])


import datetime, os
logdir = os.path.join("logs2", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

#model.fit(feature_train, label_train, epochs=200, batch_size=20, verbose=0, callbacks=[tensorboard_callback])

model.fit(inputList, resultList, epochs=200, batch_size=20, verbose=0, callbacks=[tensorboard_callback])
scores = model.evaluate(inputList, resultList)
print(type(model.metrics_names))
print(model.metrics_names)
print(model.metrics_names[1], scores[1])
print(model.metrics_names[0], scores[0])
print("after training, model coef:", output_layer.get_weights()[0])
print("after training, model intercept:", output_layer.get_weights()[1])
