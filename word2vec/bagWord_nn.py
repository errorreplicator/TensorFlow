import tensorflow as tf
import pickle
import numpy as np

open_pickle = open('./Data/bow_data.pkl','rb')
train_x, train_y, test_x, test_y = pickle.load(open_pickle)

train_x = np.array(list(train_x))
train_y = np.array(list(train_y))
test_x = np.array(list(test_x))
test_y = np.array(list(test_y))


hidden_l1 = 500
hidden_l2 = 500
hidden_l3 = 500
out_l = 2
batch_size = 100
hm_epoch = 10

X = tf.placeholder(tf.float32)#,shape=[None,len(test_x[0])])#,shape=[None,len(test_x[0])])
Y = tf.placeholder(tf.float32)#,shape=[None,2]) #Shape ??

def neural_net_model(data):

    h1_neurons = {'weights':tf.Variable(tf.random_normal([len(test_x[0]),hidden_l1])),
                  'bias':tf.Variable(tf.random_normal([hidden_l1]))}
    h2_neurons = {'weights':tf.Variable(tf.random_normal([hidden_l1,hidden_l2])),
                  'bias':tf.Variable(tf.random_normal([hidden_l2]))}
    h3_neurons = {'weights':tf.Variable(tf.random_normal([hidden_l2,hidden_l3])),
                  'bias':tf.Variable(tf.random_normal([hidden_l3]))}

    out_layer = {'weights':tf.Variable(tf.random_normal([hidden_l2,out_l])),
                 'bias':tf.Variable(tf.random_normal([out_l]))} #nawiasy kwadratowe ??

    l1 = tf.add(tf.matmul(data,h1_neurons['weights']),h1_neurons['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,h2_neurons['weights']),h2_neurons['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,h3_neurons['weights']),h3_neurons['bias'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,out_layer['weights']) + out_layer['bias']

    return (output)

def train_nn(x):
    prediction = neural_net_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epoch):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = batch_size
                batch_x = train_x[start:end]
                batch_y = train_y[start:end]
                _, c = sess.run([optimizer,cost],feed_dict={X:batch_x,Y:batch_y})
                epoch_loss += c
                i +=batch_size
                print('Epoch', epoch + 1, 'completed out of', hm_epoch, 'loss:', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        testing_x = np.array(test_x)
        testing_y = np.array(test_y)
        print('TU JESTEM',testing_x.shape)
        print(testing_y.shape)
        print('Accuracy:', accuracy.eval({X: testing_x, Y:testing_y}))

train_nn(X)