import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import to_categorical as one_hot
from sklearn.metrics import accuracy_score, confusion_matrix

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

with tf.device('/gpu:0'):
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')

    filt_size = [3, 3]
    train_labels_ohe = one_hot(Y_train)
    X_train = np.array(X_train, dtype=float) / 255.0 #rescale the data from 0-255 to 0-1
    X_test = np.array(X_test, dtype=float) / 255.0

    model = keras.models.Sequential()

    model.add(keras.layers.Reshape([32, 32, 3]))
    model.add(keras.layers.Conv2D(32, filt_size, padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(64, filt_size, activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Conv2D(128, filt_size, padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(128, filt_size, activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dense(100, activation='softmax'))


    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0001, decay=1e-6),
                  metrics=['accuracy'])

    history = model.fit(X_train, train_labels_ohe, epochs=60, batch_size=128, validation_split=0.1) #set aside 10% of the data for validation purpose

Y_pred = model.predict(X_test)

pred_cnn = []
for p in Y_pred:
    pred_cnn.append(int(np.argmax(p)))

print(accuracy_score(Y_test, pred_cnn))
print(confusion_matrix(Y_test, pred_cnn))