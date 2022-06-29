from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = keras.utils.to_categorical(y_train_full[:5000]), \
                   keras.utils.to_categorical(y_train_full[5000:])
X_test = X_test / 255.0
print(X_test.shape)

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape = [28, 28]))
model.add(keras.layers.Dense(300, activation = 'relu'))
model.add(keras.layers.Dense(100, activation = 'relu'))
model.add(keras.layers.Dense(10, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = keras.optimizers.SGD(learning_rate = 3e-1),
              metrics = ['accuracy'])

history = model.fit(X_train, y_train, epochs = 30, validation_data = (X_valid, y_valid))
pd.DataFrame(history.history).plot(figsize = (8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

print(model.summary())
print(X_test.shape)

print(model.evaluate(X_test, keras.utils.to_categorical(y_test)))
