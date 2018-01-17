import config
import os,random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import train_batch
import test_batch
import h5py
import sklearn
from sklearn.model_selection import KFold
from keras.utils import np_utils


# hyper parameters
np.random.seed(1337)  # for reproducibility
learning_rate = 0.001
batch_size = 64
training_epochs = 30
display_step = 1

# network hyper parameters
n_input = 50176
n_classes = 2
dropout = 0.75

data_x = np.array(h5py.File("/home/gen/ota/kfold/1010224224.h5",'r').get('data_x'))
label_y = np.array(h5py.File("/home/gen/ota/kfold/1010224224.h5",'r').get('label_y'))

# define convolution
def conv2d(name, x, W, b, strides = 1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x, name=name)

# define pooling
def maxpool2d(name, x, k=2):
    return tf.nn.max_pool(x, ksize = [1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

# define normlize
def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

# network parameters
weights = {
    'wc1_1': tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.1)),
    'wc1_2': tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=0.1)),

    'wc2_1': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.1)),
    'wc2_2': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=0.1)),

    'wc3_1': tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.1)),
    'wc3_2': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=0.1)),
    'wc3_3': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=0.1)),
    'wc3_4': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=0.1)),

    'wc4_1': tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=0.1)),
    'wc4_2': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.1)),
    'wc4_3': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.1)),
    'wc4_4': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.1)),

    'wc5_1': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.1)),
    'wc5_2': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.1)),
    'wc5_3': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.1)),
    'wc5_4': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.1)),

    'wd1': tf.Variable(tf.random_normal([7*7*512, 4096], stddev=0.1)),
    'wd2': tf.Variable(tf.random_normal([4096, 4096], stddev=0.1)),
    'out': tf.Variable(tf.random_normal([4096, n_classes], stddev=0.1))
}

biases = {
    'bc1_1': tf.Variable(tf.random_normal([64], stddev=0.1)),
    'bc1_2': tf.Variable(tf.random_normal([64], stddev=0.1)),

    'bc2_1': tf.Variable(tf.random_normal([128], stddev=0.1)),
    'bc2_2': tf.Variable(tf.random_normal([128], stddev=0.1)),

    'bc3_1': tf.Variable(tf.random_normal([256], stddev=0.1)),
    'bc3_2': tf.Variable(tf.random_normal([256], stddev=0.1)),
    'bc3_3': tf.Variable(tf.random_normal([256], stddev=0.1)),
    'bc3_4': tf.Variable(tf.random_normal([256], stddev=0.1)),

    'bc4_1': tf.Variable(tf.random_normal([512], stddev=0.1)),
    'bc4_2': tf.Variable(tf.random_normal([512], stddev=0.1)),
    'bc4_3': tf.Variable(tf.random_normal([512], stddev=0.1)),
    'bc4_4': tf.Variable(tf.random_normal([512], stddev=0.1)),

    'bc5_1': tf.Variable(tf.random_normal([512], stddev=0.1)),
    'bc5_2': tf.Variable(tf.random_normal([512], stddev=0.1)),
    'bc5_3': tf.Variable(tf.random_normal([512], stddev=0.1)),
    'bc5_4': tf.Variable(tf.random_normal([512], stddev=0.1)),

    'bd1': tf.Variable(tf.random_normal([4096], stddev=0.1)),
    'bd2': tf.Variable(tf.random_normal([4096], stddev=0.1)),
    'out': tf.Variable(tf.random_normal([n_classes], stddev=0.1))
}


def alex_net(x, weights, biases, dropout):
    # reshape
    x = tf.reshape(x, shape=[-1, 224, 224, 1])

    # conv1
    # conv
    conv1_1 = conv2d('conv1_1', x, weights['wc1_1'], biases['bc1_1'])
    conv1_2 = conv2d('conv1_2', conv1_1, weights['wc1_2'], biases['bc1_2'])
    # pooling
    pool1 = maxpool2d('pool1', conv1_2, k=2)
    #drop1 = tf.nn.dropout(pool1, dropout)
    # normlize
    #norm1 = norm('norm1', drop1, lsize=4)

    # conv2
    # conv
    conv2_1 = conv2d('conv2_1', pool1, weights['wc2_1'], biases['bc2_1'])
    conv2_2 = conv2d('conv2_2', conv2_1, weights['wc2_2'], biases['bc2_2'])
    # pooling
    pool2 = maxpool2d('pool2', conv2_2, k=2)
    #drop2 = tf.nn.dropout(pool2, dropout)
    # normlize
    # norm2 = norm('norm2', drop2, lsize=4)

    # conv3
    # conv
    conv3_1 = conv2d('conv3_1', pool2, weights['wc3_1'], biases['bc3_1'])
    conv3_2 = conv2d('conv3_2', conv3_1, weights['wc3_2'], biases['bc3_2'])
    conv3_3 = conv2d('conv3_3', conv3_2, weights['wc3_3'], biases['bc3_3'])
    conv3_4 = conv2d('conv3_4', conv3_3, weights['wc3_4'], biases['bc3_4'])
    # pooling
    pool3 = maxpool2d('pool3', conv3_4, k=2)
    #drop3 = tf.nn.dropout(pool3, dropout)

    # conv4
    conv4_1 = conv2d('conv4_1', pool3, weights['wc4_1'], biases['bc4_1'])
    conv4_2 = conv2d('conv4_2', conv4_1, weights['wc4_2'], biases['bc4_2'])
    conv4_3 = conv2d('conv4_3', conv4_2, weights['wc4_3'], biases['bc4_3'])
    conv4_4 = conv2d('conv4_4', conv4_3, weights['wc4_4'], biases['bc4_4'])
    # pooling
    pool4 = maxpool2d('pool3', conv4_4, k=2)
    #drop4 = tf.nn.dropout(pool4, dropout)
    # normlize
    #norm3 = norm('norm3', conv4, lsize=4)

    
    # conv5
    conv5_1 = conv2d('conv5_1', pool4, weights['wc5_1'], biases['bc5_1'])
    conv5_2 = conv2d('conv5_2', conv5_1, weights['wc5_2'], biases['bc5_2'])
    conv5_3 = conv2d('conv5_3', conv5_2, weights['wc5_3'], biases['bc5_3'])
    conv5_4 = conv2d('conv5_4', conv5_3, weights['wc5_4'], biases['bc5_4'])
    # pooling
    pool5 = maxpool2d('pool4', conv5_4, k=2)
    # normlize
    # norm4 = norm('norm4', conv5, lsize=4)

    # fc1
    fc1 = tf.reshape(pool5, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # fc2
    fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)

    # dropout
    fc2 = tf.nn.dropout(fc2, dropout)

    # output
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out

# define placeholder
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout

# create model
pred = alex_net(x, weights, biases, keep_prob)

# define lost_func & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#cost = tf.reduce_mean(tf.square(y-pred))
#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
#score
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
y_p = tf.argmax(pred, 1)
#define initialize
init = tf.global_variables_initializer()
#print ("GRAPH READY")

#define k-fold cross validation
kf = KFold(n_splits=5, shuffle=True)

count = 0
for train_index, test_index in kf.split(data_x):
    print("%s, %s" % (train_index, test_index))
    img_rows,img_cols = config.img_row, config.img_col
    X_train, X_test = data_x[train_index], data_x[test_index]
    y_train, y_test = label_y[train_index], label_y[test_index]
    print(X_train.shape)
    print(X_test.shape)

    X_train = X_train.reshape(X_train.shape[0], img_rows * img_cols)
    X_test = X_test.reshape(X_test.shape[0], img_rows * img_cols)

    print(X_train.shape)
    print(X_test.shape)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices
    #print(y_train)
    #print(y_test)

    y_train = np_utils.to_categorical(y_train, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)

    #print(y_train.shape)
    #print(y_test)

    train = train_batch.DataSet(X_train, y_train)
    test = test_batch.DataSet(X_test, y_test)

    sess = tf.Session()
    sess.run(init)
    print ("GRAPH READY")

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(X_train.shape[0]/batch_size)
        total_batch_2 = int(X_test.shape[0]/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob:dropout})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob:1.})/total_batch

        # Display logs per epoch step
        if epoch % display_step == 0:
            avg_train = 0
            for j in range(total_batch):
                batch_xs, batch_ys = train.next_batch(batch_size)
                train_acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob:1.})
                avg_train += train_acc
            avg_train = avg_train / total_batch
            print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
            print (" Training accuracy: %.4f" % (avg_train))
            avg_test = 0
            avg_y_true = 0
            avg_y_pred = 0
            for k in range(total_batch_2):
                batch_xt, batch_yt = test.next_batch(batch_size)
                val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x: batch_xt, y: batch_yt, keep_prob:1.})
                #test_acc = sess.run(accuracy, feed_dict={x: batch_xt, y: batch_yt, keep_prob:1.})
                y_true = np.argmax(batch_yt, 1)
                avg_y_true += y_true
                avg_y_pred += y_pred
                avg_test += val_accuracy
            avg_test = avg_test / total_batch_2
            avg_y_true = avg_y_true / total_batch_2
            avg_y_pred = avg_y_pred / total_batch_2
            print (" Test accuracy: %.4f" % (avg_test))
            print (" Precision: %.4f" % (sklearn.metrics.precision_score(avg_y_true, avg_y_pred)))
            print (" Recall: %.4f" % (sklearn.metrics.recall_score(avg_y_true, avg_y_pred)))
            print (" f1_score: %.4f" % (sklearn.metrics.f1_score(avg_y_true, avg_y_pred)))
    count += 1
    print ("OPTIMIZATION FINISHED")
    print (count)
    #val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x:})
