import os
from urllib.request import urlopen
import zipfile
import tensorflow as tf
import scipy
import pickle
from tensorflow.keras.utils import to_categorical as one_hot
from tensorflow import keras
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#-----------------------------------------------------------------------------
#download the pre-trained model (GoogLeNet) and import the model into a tensorflow graph
data_url = "http://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip"
data_dir = './inception_5h/'
file_path = os.path.join(data_dir, 'inception5h.zip')

if not os.path.exists(file_path):
    # Check if the download directory exists, otherwise create it.
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    # Download
    with open(file_path, "wb") as local_file:
        local_file.write(urlopen(data_url).read())
    # Extract
    zipfile.ZipFile(file_path, mode="r").extractall(data_dir)

path = os.path.join(data_dir, "tensorflow_inception_graph.pb")
with tf.gfile.FastGFile(path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.device('/gpu:0'): #run the task on GPU
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')

    #----------------------------------------------------------------------------------
    #run the training images through the GoogLeNet
    googlenet_input = sess.graph.get_tensor_by_name("input:0")
    googlenet_output = sess.graph.get_tensor_by_name("avgpool0:0")
    BATCH_SIZE = 200
    latent_vec = np.zeros([1, 1, 1, 1024])#initialize an ndarray to store the output vectors

    # rescale the images and run them through the inception in 200-image batches
    #run the train images through GoogLeNet
    print('Creating latent vectors for train images from GoogLeNet...')
    j = 0
    for _ in range(250): #BATCH_SIZE * 250 = 50000
        if _ % 10 == 0:
            print(_, 'out of 250 batches')

        dr_input = scipy.ndimage.zoom(X_train[j:j + BATCH_SIZE, :, :, :], zoom=(1, 224 / 32, 224 / 32, 1), order=1) #input to GooogLeNet needs to resized to meet its requirements for input dimensions
        googlenet_out_tensor = sess.run(googlenet_output, feed_dict={googlenet_input: dr_input})
        latent_vec = np.vstack((latent_vec, googlenet_out_tensor))
        j += BATCH_SIZE

    # remove the first element which is an all-zero array
    latent_vec = latent_vec[1:, :, :, :]
    # save the output vectors to a local machine
    pickle.dump(latent_vec[1:, :, :, :], open('inception_output.p', 'wb'))

    # repeat the same steps for the validation images
    googlenet_input = sess.graph.get_tensor_by_name("input:0")
    googlenet_output = sess.graph.get_tensor_by_name("avgpool0:0")
    BATCH_SIZE = 200
    latent_vec_val = np.zeros([1, 1, 1, 1024])  # initialize an ndarray to store the output vectors

    #run the test images through GoogLeNet
    print('Creating latant vectors for train images from GoogLeNet...')
    j = 0
    for _ in range(50):
        if _ % 10 == 0:
            print(_, 'out of 50 batches')

        dr_input = scipy.ndimage.zoom(X_test[j:j + BATCH_SIZE, :, :, :], zoom=(1, 224 / 32, 224 / 32, 1), order=1)
        googlenet_out_tensor = sess.run(googlenet_output, feed_dict={googlenet_input: dr_input})
        latent_vec_val = np.vstack((latent_vec_val, googlenet_out_tensor))
        j += BATCH_SIZE

    #remove the first element which is an all-zero array
    latent_vec_val = latent_vec_val[1:, :, :, :]

    #save the output vectors to a local machine
    pickle.dump(latent_vec_val, open('validation_output.p', 'wb'))

    #--------------------------------------------------------------------------------------------------------
    #feed the output from GoogLeNet to a MLP model
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
    history = transfer_model.fit(latent_vec, train_labels_ohe, epochs=10, batch_size=250)

#make prediction
prediction_transfer = transfer_model.predict(latent_vec_val)

predcls_transfer = []
for p in prediction_transfer:
    predcls_transfer.append(int(np.argmax(p)))

print(accuracy_score(Y_test, predcls_transfer))
print(confusion_matrix(Y_test, predcls_transfer))