import tensorflow as tf
import numpy as np # Matrix and vector computation package
import pandas as pd
import os

data = pd.read_csv('D:/Anaconda3/envs/HR/project/breast_cancer_project/input/corr.csv')
data = data.as_matrix()
#np.random.shuffle(data)
#np.savetxt("shuffled.csv", data, delimiter=",")

training, test = data[:400,:], data[400:,:]
x_data = training[:,:6]
y_data = training[:,6:]
test_x_data = test[:,:6]
test_y_data = test[:,6:]



# Parameters
learning_late = 0.001
training_epochs = 20
batch_szie = 50
display_step = 1

# tf Graph input
X = tf.placeholder("float", [None, 6])
Y = tf.placeholder("float", [None, 2])

#store layers weight & bias
W1 = tf.get_variable("W1", shape=[6,20], initializer=tf.contrib.layers.xavier_initializer((6,20)))
W2 = tf.get_variable("W2", shape=[20,20], initializer=tf.contrib.layers.xavier_initializer((20,20)))
W3 = tf.get_variable("W3", shape=[20,20], initializer=tf.contrib.layers.xavier_initializer((20,20)))
W4 = tf.get_variable("W4", shape=[20,20], initializer=tf.contrib.layers.xavier_initializer((20,20)))
W5 = tf.get_variable("W5", shape=[20,20], initializer=tf.contrib.layers.xavier_initializer((20,20)))
#W6 = tf.get_variable("W6", shape=[256,256], initializer=tf.contrib.layers.xavier_initializer((256,256)))
#W7 = tf.get_variable("W7", shape=[256,256], initializer=tf.contrib.layers.xavier_initializer((256,256)))
#W8 = tf.get_variable("W8", shape=[256,256], initializer=tf.contrib.layers.xavier_initializer((256,256)))
W9 = tf.get_variable("W9", shape=[20,2], initializer=tf.contrib.layers.xavier_initializer((20,2)))

tf.summary.histogram("w1", W1)
tf.summary.histogram("w2", W2)
tf.summary.histogram("w3", W3)
tf.summary.histogram("w4", W4)
tf.summary.histogram("w5", W5)
#tf.summary.histogram("w6", W6)
#tf.summary.histogram("w7", W7)
#tf.summary.histogram("w8", W8)
tf.summary.histogram("w9", W9)



B1 = tf.Variable(tf.random_normal([20]))
B2 = tf.Variable(tf.random_normal([20]))
B3 = tf.Variable(tf.random_normal([20]))
B4 = tf.Variable(tf.random_normal([20]))
B5 = tf.Variable(tf.random_normal([20]))
#B6 = tf.Variable(tf.random_normal([256]))
#B7 = tf.Variable(tf.random_normal([256]))
#B8 = tf.Variable(tf.random_normal([256]))
B9 = tf.Variable(tf.random_normal([2]))

#construct model    with xavier_initializer and  more deep and  dropout
dropout_rate = tf.placeholder("float")
with tf.name_scope("layer1"):
    _L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1))
    L1 = tf.nn.dropout(_L1, dropout_rate)
with tf.name_scope("layer2"):
    _L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), B2)) #hidden layer with relu activation
    L2 = tf.nn.dropout(_L2, dropout_rate)

with tf.name_scope("layer3"):
    _L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), B3)) #hidden layer with relu activation
    L3 = tf.nn.dropout(_L3, dropout_rate)

with tf.name_scope("layer4"):
    _L4 = tf.nn.relu(tf.add(tf.matmul(L3, W4), B4)) #hidden layer with relu activation
    L4 = tf.nn.dropout(_L4, dropout_rate)

with tf.name_scope("layer5"):
    _L5 = tf.nn.relu(tf.add(tf.matmul(L4, W5), B5)) #hidden layer with relu activation
    L5 = tf.nn.dropout(_L5, dropout_rate)
"""
with tf.name_scope("layer6"):
    _L6 = tf.nn.relu(tf.add(tf.matmul(L5, W6), B6)) #hidden layer with relu activation
    L6 = tf.nn.dropout(_L6, dropout_rate)
with tf.name_scope("layer7"):
    _L7 = tf.nn.relu(tf.add(tf.matmul(L6, W7), B7)) #hidden layer with relu activation
    L7 = tf.nn.dropout(_L7, dropout_rate)
with tf.name_scope("layer8"):
    _L8 = tf.nn.relu(tf.add(tf.matmul(L7, W8), B8)) #hidden layer with relu activation
    L8 = tf.nn.dropout(_L8, dropout_rate)
"""
hypothesis = tf.add(tf.matmul(L5, W9), B9) #no need to use softmax here



with tf.name_scope("cost"):
    #define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y)) #softmax loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_late).minimize(cost)
    tf.summary.scalar("cost", cost)

with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    acc_op = tf.reduce_mean(tf.cast(correct_pred, "float"))
    tf.summary.scalar("accuracy", acc_op)

#initializing the var
init = tf.initialize_all_variables()

# saving model
saver = tf.train.Saver()
model_path = './models/model13/'
if not os.path.exists(model_path):
    os.makedirs(model_path)



#lauch the graph
with tf.Session() as sess:
    writer = tf.summary.FileWriter("./logds/nn_logs/model13", sess.graph)  # for 1.0
    merged = tf.summary.merge_all()
    sess.run(init)

    #training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        #total_batch = int(mnist.train.num_examples/batch_szie)
        total_batch = 100
        print("total_batch:", total_batch)
        #Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = x_data, y_data
            #fit training using batch data
            sess.run(optimizer, feed_dict={X:batch_xs, Y:batch_ys, dropout_rate:1.0})
            #compute average loss
            avg_cost += sess.run(cost, feed_dict={X:batch_xs, Y:batch_ys, dropout_rate:1.0})/total_batch
        #display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

        summary, acc = sess.run([merged, acc_op], feed_dict={X: test_x_data, Y: test_y_data, dropout_rate: 1.0})
        writer.add_summary(summary, epoch)  # Write summary
        print("Epoch:", '%04d' % (epoch+1), "Test Accuracy :", acc)
    writer.close()
    print("optimization finished!")

    #saving model
    save_path = saver.save(sess, model_path)
    print("Model saved in :", save_path)

"""
    # one case
    a = sess.run(hypothesis, feed_dict={X:[[72, 76, 100, 80]], dropout_rate:1.0}) #A
    b = tf.nn.softmax(a)
    print(a, sess.run(tf.arg_max(b, 1)))

    a = sess.run(hypothesis, feed_dict={X: [[39, 78, 32, 66]], dropout_rate: 1.0}) #B
    b = tf.nn.softmax(a)
    print(a, sess.run(tf.arg_max(b, 1)))

    a = sess.run(hypothesis, feed_dict={X:[[11, 13, 11, 62]], dropout_rate:1.0}) #D
    b = tf.nn.softmax(a)
    print(a, sess.run(tf.arg_max(b, 1)))



# Keep Training by Loading Existing Models
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver.restore(sess, model_path)
    print("Model loaded from :", model_path)

    a = sess.run(hypothesis, feed_dict={X: [[0.38, 0.53, 2, 157, 3, 0, 0, 0, 0]], dropout_rate: 1.0})  # B
    b = tf.nn.softmax(a)
    print(a, sess.run(tf.arg_max(b, 1)))

    #saving model
    save_path = saver.save(sess, model_path)
    print("Model saved in :", save_path)
"""