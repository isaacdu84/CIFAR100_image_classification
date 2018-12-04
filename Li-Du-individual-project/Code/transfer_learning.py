import tensorflow as tf
import pickle
from tensorflow.keras.utils import to_categorical as one_hot
from tensorflow import keras
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

with tf.device('/gpu:0'): #run the task on GPU
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')

    #--------------------------------------------------------------------------------------------------------
    #feed the output from GoogLeNet to fully connected layers
    latent_vec = pickle.load(open('inception_output.p', 'rb'))
    latent_vec_val = pickle.load(open('validation_output.p', 'rb'))

    #one-hot encode the labels so that it is compatible with the softmax activation function
    train_labels_ohe = one_hot(Y_train)
    filt_size = [3, 3]

    transfer_model = keras.models.Sequential()
    transfer_model.add(keras.layers.Flatten())
    transfer_model.add(keras.layers.Dense(1024, activation='relu'))
    transfer_model.add(keras.layers.Dropout(0.5))
    transfer_model.add(keras.layers.Dense(1024, activation='relu'))#use_bias=True, kernel_initializer=keras.initializers.TruncatedNormal(stddev=img_size_flat**-0.5))
    transfer_model.add(keras.layers.Dense(100, activation='softmax'))
    transfer_model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=0.0001, decay=1e-6),
                  metrics=['accuracy'])
    history = transfer_model.fit(latent_vec, train_labels_ohe, epochs=30, batch_size=250, validation_split=0.2)

#make prediction
prediction_transfer = transfer_model.predict(latent_vec_val)

predcls_transfer = []
for p in prediction_transfer:
    predcls_transfer.append(int(np.argmax(p)))

print(accuracy_score(Y_test, predcls_transfer))
print(confusion_matrix(Y_test, predcls_transfer))

#create the the actual labels
CIFAR100_LABELS_LIST = np.asarray([
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
])

cm = confusion_matrix(Y_test, predcls_transfer)
accu_by_class = cm.diagonal()/100
df_accu_by_class = pd.DataFrame({'Labels': CIFAR100_LABELS_LIST, 'Accu': accu_by_class})
df_accu_by_class.sort_values(by='Accu',ascending=False,inplace=True)
#--------------------------------------------------------------------------------------------------------------
#visualization

#visualization of the training on fully connected layers
sns.set_style('darkgrid')
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['acc'], label='Training')
plt.plot(history.history['val_acc'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

cm = pd.DataFrame(cm)
plt.figure(figsize=(15,12))
sns.heatmap(cm)
plt.yticks(rotation=0, fontsize=7.5)
plt.xticks(rotation=315, fontsize=7.5)
plt.show()

plt.figure(figsize=(15, 8))
sns.barplot('Labels', 'Accu', data=df_accu_by_class)
plt.show()