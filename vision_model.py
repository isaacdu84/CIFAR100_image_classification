import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical as one_hot
from sklearn.metrics import accuracy_score, confusion_matrix

#import the raw data
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
#train_labels_ohe = one_hot(Y_train, 100)
#test_labels_ohe = one_hot(Y_test, 100)

#---------------------------------------------------------------------------------------------
def convert_images(raw):

    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images


#define a function to calculate the delta between two pictures
images = tf.placeholder(shape = (50000, 32, 32, 3), dtype = np.float32)
base = tf.placeholder(shape = (32, 32, 3), dtype = np.float32)
def delta_func(img_graph, base_graph):
    #img_graph = images
    #base_graph = base
    deltaR = tf.squared_difference(img_graph[:, :, :, 0], base_graph[:, :, 0], name='deltaR')
    deltaG = tf.squared_difference(img_graph[:, :, :, 1], base_graph[:, :, 1], name='deltaG')
    deltaB = tf.squared_difference(img_graph[:, :, :, 2], base_graph[:, :, 2], name='deltaB')
    Rbar = 0.5 * tf.add(img_graph[:, :, :, 0], base_graph[:,:,0], name = 'rbar')
    deltaC = tf.sqrt(tf.add_n([2 * deltaR, 4 * deltaG, 3 * deltaB, Rbar * (deltaR - deltaB)], name='deltaC'), name='deltaCsqrt')
    deltaC_mean_h = tf.reduce_mean(deltaC, axis=1)
    deltaC_mean = tf.reduce_mean(deltaC_mean_h, axis=1)
    return deltaC_mean

#include the variable initialization process into a function
def reset_vars():
    sess.run(tf.global_variables_initializer())

#---------------------------------------------------------------------------------------------
X_train = np.array(X_train, dtype=float) / 255.0
X_test = np.array(X_test, dtype=float) / 255.0


#---------------------------------------------------------------------------------------------
sess = tf.Session()

#reorganize the images so that they follow the structure defined in "image_class"
image_class = np.zeros((100,500,32,32,3)) #100 classes 500 images in each classes
class_index = [0 for i in range(100)]
for i in range(50000):
    labeli = int(Y_train[i])
    image_class[labeli,class_index[labeli], :, :, :] = X_train[i]
    class_index[labeli] += 1

#set up the variables and error functions
#the idea is to train a "typical" image for each image class by creating an image that minimize the delta to all the
#images in a particular class
lr = 20
typical = tf.Variable(tf.random_uniform([32, 32, 3], dtype=tf.float64))
images = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float64)
error = tf.reduce_mean(tf.square(delta_func(images, typical)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
train_typ = optimizer.minimize(error)


typicals = []
ers = []
BATCH_SIZE = 20
for i in range(100):
    print('i =', i)
    reset_vars()
    er = sess.run(error, feed_dict={images: image_class[i, :, :, :, :]})
    ers.append([er])
    for _ in range(60):
        j = np.random.choice(500, BATCH_SIZE, replace=False)
        sess.run(train_typ, feed_dict={images: image_class[i, j, :, :, :]})
        er = sess.run(error, feed_dict={images: image_class[i, :, :, :, :]})
        ers[-1].append(er)
    typicali = sess.run(typical)
    typicals.append(typicali)
typicals = np.hstack([typicals]) #now this should include 100 "typical" images corresponding to each class

#----------------------------------------------------------------------------------------------
#Next step would be implement a softmax model that calculate the probability of one image being predicted as each of the classes

lr_softmax = 0.5

x = tf.placeholder(dtype=tf.float32, shape=[None, 100], name='pixels')#100 classes
y_label = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='label')

W = tf.Variable(tf.zeros([100, 100]), name="weights")
b = tf.Variable(tf.zeros([100]), name="biases")
y = tf.matmul(x, W) + b

#calculate the distance to each typical image
images = tf.placeholder(dtype=tf.float64, shape=(None, 32, 32, 3))
base = tf.placeholder(dtype=tf.float64, shape=(32, 32, 3))
z = delta_func(images, base)
dist = []
for imgi in X_train:
    dist.append(sess.run(z, {images: typicals, base:imgi}))

dist_array = np.array(dist, dtype=np.float64)
train_labels_ohe = tf.one_hot(y_label, 100) #one-hot encode the labels of the training set because the softmax requires the labels to be one-hot encoded

#define the train function to run
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=train_labels_ohe))
train = tf.train.GradientDescentOptimizer(lr_softmax).minimize(loss)

reset_vars()

for i in range(2000):
    sess.run(train, feed_dict={x: dist_array, y_label: Y_train})

    if i % 100 == 0:
        print('Iteration:', i)

#make predictions
images = tf.placeholder(dtype=tf.float64, shape=(None, 32, 32, 3))
base = tf.placeholder(dtype=tf.float64, shape=(32, 32, 3))
z = delta_func(images, base)
dist_val = []
for imgi in X_test:
    dist_val.append(sess.run(z, {images: typicals, base:imgi}))
dist_val_array = np.array(dist_val, dtype=np.float32)

x_val = tf.placeholder(dtype=tf.float32, shape=[None, 100], name='pixels')
y_pred = tf.matmul(x_val, W) + b

prob_score = sess.run(y_pred, feed_dict={x_val: dist_val_array})

pred_softmax = []
for img in prob_score:
    pred_softmax.append(int(np.argmax(img)))

print(accuracy_score(Y_test, pred_softmax))
print(confusion_matrix(Y_test, pred_softmax))
