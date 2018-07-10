import config
import sys
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
from PIL import Image

#command line parameters
args = sys.argv

# hyper parameters
np.random.seed(1337)  # for reproducibility
learning_rate = 0.001
batch_size = 32
training_epochs = 40
display_step = 1
num_channel = 1

# network hyper parameters
n_classes = 2
dropout = 0.75
n_input = config.img_row * config.img_col

data_x = np.array(h5py.File("/home/s1411506/ota/kfold/CGNR.h5",'r').get('data_x'))
label_y = np.array(h5py.File("/home/s1411506/ota/kfold/CGNR.h5",'r').get('label_y'))

# define convolution
def conv2d(name, x, W, b, strides = 1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return x

def activation_relu(x):
    return tf.nn.relu(x)

# define pooling
def maxpool2d(name, x, k=2):
    return tf.nn.max_pool(x, ksize = [1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

# define normlize
#def norm(name, l_input, lsize=4):
#    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

# network parameters
weights = {
    'wc1_1': tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.1)),
    'wc1_2': tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.1)),    
    'wc2_1': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.1)),
    'wd1': tf.Variable(tf.random_normal([14*14*128, 256], stddev=0.1)),
    'wd2': tf.Variable(tf.random_normal([256, 512], stddev=0.1)),
    'out': tf.Variable(tf.random_normal([512, n_classes], stddev=0.1))
}
 
biases = {
    'bc1_1': tf.Variable(tf.random_normal([32], stddev=0.1)),
    'bc1_2': tf.Variable(tf.random_normal([64], stddev=0.1)),    
    'bc2_1': tf.Variable(tf.random_normal([128], stddev=0.1)),
    'bd1': tf.Variable(tf.random_normal([256], stddev=0.1)),
    'bd2': tf.Variable(tf.random_normal([512], stddev=0.1)),
    'out': tf.Variable(tf.random_normal([n_classes], stddev=0.1))
}


def alex_net(x, weights, biases, dropout):
    # reshape
    x = tf.reshape(x, shape=[-1, config.img_row, config.img_col, num_channel])

    #conv & pool1
    conv1_1 = conv2d('conv1_1', x, weights['wc1_1'], biases['bc1_1'])
    conv1_1 = activation_relu(conv1_1)
    pool1 = maxpool2d('pool1', conv1_1, k=2)

    #conv & pool2
    conv1_2 = conv2d('conv1_2', pool1, weights['wc1_2'], biases['bc1_2'])
    conv1_2 = activation_relu(conv1_2)
    pool2 = maxpool2d('pool2', conv1_2, k=2)
    drop1 = tf.nn.dropout(pool2, dropout)

    # conv & pool3
    conv2_1 = conv2d('conv2_1', drop1, weights['wc2_1'], biases['bc2_1'])
    conv2_1 = activation_relu(conv2_1)
    pool3 = maxpool2d('pool3', conv2_1, k=2)
    drop2 = tf.nn.dropout(pool3, dropout)

    # fc1
    fc1 = tf.reshape(drop2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = activation_relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    # fc2
    fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
    fc2 = activation_relu(fc2)
    #fc2 = tf.nn.dropout(fc2, dropout)

    # output
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])

    return out

# define placeholder
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) 

# create model
pred = alex_net(x, weights, biases, keep_prob)

# define lost_func & optimizer
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
y_p = tf.argmax(pred, 1)

#define initialize
init = tf.global_variables_initializer()

#define k-fold cross validation
kf = KFold(n_splits=5, shuffle=True)

sess = tf.Session()
count = 0
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('./')
if ckpt and len(args) == 2:
    last_model = ckpt.model_checkpoint_path
    saver.restore(sess, last_model)
    new_img = Image.open(args[1]).convert('L')
    new_img = new_img.resize((config.img_row,config.img_col))
    #new_img = 1.0 - np.asarray(new_img, dtype="float32") / 255
    new_img = np.asarray(new_img, dtype="float32") / 255
    new_img = new_img.reshape((1, config.img_row * config.img_col ) )    
    prediction = y_p
    print(args[1]+ ":%g"%prediction.eval(feed_dict={x: new_img, keep_prob: 1.0}, session=sess))
else:
    loss1 = []
    loss2 = []
    percentage = []
    for train_index, test_index in kf.split(data_x):
        print("%s, %s" % (train_index, test_index))
        img_rows,img_cols = config.img_row, config.img_col
        X_train, X_test = data_x[train_index], data_x[test_index]
        y_train, y_test = label_y[train_index], label_y[test_index]
        #print(X_train.shape)
        #print(X_test.shape)

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
        
        #sess = tf.Session()
        acc_train_v = []
        acc_test_v = []
        sess.run(init)
        print ("GRAPH READY")
        for epoch in range(training_epochs):
            avg_cost = 0.
            avg_cost2 = 0.
            total_batch = int(X_train.shape[0]/batch_size)
            total_batch_2 = int(X_test.shape[0]/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs,batch_ys = train.next_batch(batch_size)
                batch_us,batch_vs = test.next_batch(batch_size)
                # Fit training using batch data
                sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob:dropout})
                # Compute average loss
                avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob:1.})/total_batch
                avg_cost2 += sess.run(cost, feed_dict={x: batch_us, y:batch_vs, keep_prob:1.})/total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                avg_train = 0
                for j in range(total_batch):
                    batch_xs, batch_ys = train.next_batch(batch_size)
                    train_acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob:1.})
                    avg_train += train_acc
                avg_train = avg_train / total_batch
                print ("Epoch: %03d/%03d" % (epoch, training_epochs))
                print("Training cost: %.9f" %(avg_cost))
                print("Test cost: %.9f" % (avg_cost2))
                print (" Training accuracy: %.4f" % (avg_train))
                loss1.append(avg_cost)
                loss2.append(avg_cost2)
                avg_test = 0
                avg_y_true = 0
                avg_y_pred = 0
                for k in range(total_batch_2):
                    batch_xt, batch_yt = test.next_batch(batch_size)
                    val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x: batch_xt, y: batch_yt, keep_prob:1.})
                    test_acc = sess.run(accuracy, feed_dict={x: batch_xt, y: batch_yt, keep_prob:1.})
                    y_true = np.argmax(batch_yt, 1)
                    avg_y_true += y_true
                    avg_y_pred += y_pred
                    avg_test += val_accuracy
                avg_test = avg_test / total_batch_2
                avg_y_true = avg_y_true / total_batch_2
                avg_y_pred = avg_y_pred / total_batch_2
                print (" Test accuracy: %.4f" % (avg_test))
                acc_train_v.append(train_acc)
                acc_test_v.append(test_acc)
                """
                print (" Precision: %.4f" % (sklearn.metrics.precision_score(avg_y_true, avg_y_pred)))
                print (" Recall: %.4f" % (sklearn.metrics.recall_score(avg_y_true, avg_y_pred)))
                print (" f1_score: %.4f" % (sklearn.metrics.f1_score(avg_y_true, avg_y_pred)))
                """
                #percentage.append(avg_test)
        count += 1

        """
        # plot the graph of loss
        plt.plot(range(0,training_epochs),loss1,'r',label='Training')
        plt.plot(range(0,training_epochs),loss2,'b',label='Test')
        plt.legend(loc='upper right',prop={'size':11})
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

        #plot the graph of accuracy
        plt.plot(range(0,training_epochs),acc_train_v,'r',label='Training')
        plt.plot(range(0,training_epochs),acc_test_v,'b',label='Test')
        plt.legend(loc='upper right',prop={'size':11})
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()
        """
        
        saver.save(sess, "./NRmodel.ckpt")
        print ("OPTIMIZATION FINISHED")
        break
    #val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x:})
