import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical as one_hot
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

#import data from the built-in package
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
train_labels_ohe = one_hot(Y_train)
test_labels_ohe = one_hot(Y_test)

#multi-layer perceptron
mlp_model = keras.models.Sequential()
hidden_size = 128
N_PIXELS = 32 * 32 * 3

mlp_model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))

mlp_model.add(keras.layers.Dense(hidden_size, activation='sigmoid', use_bias=True,
                             kernel_initializer=keras.initializers.TruncatedNormal(stddev=N_PIXELS ** -0.5)))
mlp_model.add(keras.layers.Dense(hidden_size, activation='sigmoid', use_bias=True,
                             kernel_initializer=keras.initializers.TruncatedNormal(stddev=N_PIXELS ** -0.5)))
mlp_model.add(keras.layers.Dropout(0.25))
mlp_model.add(keras.layers.Dense(100, activation='softmax',
                                 kernel_initializer=keras.initializers.TruncatedNormal(stddev=N_PIXELS ** -0.5)))
mlp_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=0.1),metrics=['accuracy'])
history = mlp_model.fit(X_train, train_labels_ohe, epochs=100, batch_size=200)

#predict labels
mlp_predictions = mlp_model.predict(X_test)
precls_mlp = []
for p in mlp_predictions:
    precls_mlp.append(int(np.argmax(p)))

precls_mlp = np.array(precls_mlp)

print(accuracy_score(Y_test, precls_mlp))
print(confusion_matrix(Y_test, precls_mlp))