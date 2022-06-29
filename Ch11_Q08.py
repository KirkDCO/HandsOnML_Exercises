import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# get the data
cifar10 = keras.datasets.cifar10
(X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.0

# set up learning rate test function
K = keras.backend

class ExponentialLearningRate(keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []
    def on_batch_end(self, batch, logs):
        self.rates.append(K.get_value(self.model.optimizer.learning_rate))
        self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.learning_rate, self.model.optimizer.learning_rate * self.factor)

expon_lr = ExponentialLearningRate(factor=1.005)

# set seeds
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

# build the model
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape = [32, 32, 3]))
for _ in range(20):
    model.add(keras.layers.Dense(100, activation = 'selu', kernel_initializer="lecun_normal"))
model.add(keras.layers.Dense(10, activation = 'softmax'))

X_means = X_train.mean(axis=0)
X_stds = X_train.std(axis=0)
X_train_scaled = (X_train - X_means) / X_stds
X_valid_scaled = (X_valid - X_means) / X_stds
X_test_scaled = (X_test - X_means) / X_stds

# # fit the model with callback option to test learning rate
# model.compile(loss = 'sparse_categorical_crossentropy',
#               optimizer = keras.optimizers.Nadam(lr = 1e-7),
#               metrics = ['accuracy'])
#
# history = model.fit(X_train, y_train, epochs=3,
#                     validation_data=(X_valid, y_valid),
#                     callbacks=[expon_lr])
#
# # see what is happening with the learning rate
# plt.plot(expon_lr.rates, expon_lr.losses)
# plt.gca().set_xscale('log')
# plt.hlines(min(expon_lr.losses), min(expon_lr.rates), max(expon_lr.rates))
# plt.axis([min(expon_lr.rates), max(expon_lr.rates), 0, expon_lr.losses[0]])
# plt.grid()
# plt.xlabel("Learning rate")
# plt.ylabel("Loss")
# plt.show()

# from plot 1e-3 is where loss bottoms out and starts to creep up, then jumps -> try 5e-4
# with batch normalization it jumped at .01 -> try 5e-3
# selu -> 5e-5

# compoile with new learning rate
model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = keras.optimizers.Nadam(learning_rate = 5e-5),
              metrics = ['accuracy'])

early_stopping_cb = keras.callbacks.EarlyStopping(patience = 20, restore_best_weights = True)
history = model.fit(X_train, y_train, epochs = 100,
                    validation_data = (X_valid, y_valid),
                    callbacks = [early_stopping_cb])

pd.DataFrame(history.history).plot(figsize = (8, 10))
plt.grid(True)
plt.gca().set_ylim(0, 2)
plt.show()

print(model.evaluate(X_train, y_train))
print(model.evaluate(X_valid, y_valid))
print(model.evaluate(X_test, y_test))
