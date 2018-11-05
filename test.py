from tensorflow.examples.tutorials import mnist
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
# print(x_train[0])

X = tf.placeholder(tf.float32,[None,784])
Y = tf.placeholder(tf.float32,[None,784])

hidden_l1 = 500
hidden_l2 = 500
hidden_l3 = 500

out_class = 10
batch_size = 100
epoch = 50

def neural_net(data):

    Layer_hid1 = {'weight': tf.Variable(tf.random_normal(tf.float32,[784,hidden_l1])),
                  'biase': tf.random_normal(tf.Variable(tf.float32,hidden_l1))}
    # Bias_hid1 = {'bias':tf.Variable(tf.float32,hidden_l1)}
    Layer_hid2 = {'weight': tf.Variable(tf.random_normal(tf.float32, [hidden_l1,hidden_l2])),
                  'biase': tf.random_normal(tf.Variable(tf.float32, hidden_l2))}
    Layer_hid3 = {'weight': tf.Variable(tf.random_normal(tf.float32, [hidden_l2, hidden_l3])),
                  'biase': tf.random_normal(tf.Variable(tf.float32, hidden_l3))}
    Layer_out = {'weight': tf.Variable(tf.random_normal(tf.float32, [hidden_l3,out_class])),
                  'biase': tf.random_normal(tf.Variable(tf.float32, out_class))}

    l1_out = tf.add(tf.matmul(data,Layer_hid1['weight']),Layer_hid1['biase'])
    l1_out = tf.nn.relu(l1_out)

    l2_out = tf.add(tf.matmul(l1_out,Layer_hid2['weight']),Layer_hid2['biase'])
    l2_out = tf.nn.relu(l2_out)

    l3_out = tf.add(tf.matmul(l2_out, Layer_hid3['weight']), Layer_hid3['biase'])
    l3_out = tf.nn.relu(l3_out)

    out = tf.add(tf.matmul(l3_out, Layer_out['weight']), Layer_out['biase'])

    return (out)

def train_nn(x):
    prediction = neural_net(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        epoch_loss = 0
        for index in range(epoch):
            # for mnist.tr
            _, c = sess.run([optimizer,cost],feed_dict={X:tf.convert_to_tensor(x_train,tf.float32),Y:tf.convert_to_tensor(y_train,tf.float32)})
            epoch_loss+=c
            print(index,'completed out of:',epoch,'with loss:',epoch_loss)

        correctnes = tf.equal(tf.arg_max(prediction,1),tf.arg_max(Y,1))
        accuracy = tf.reduce_mean(tf.cast(correctnes,tf.float32))
        # print('Accuracy',accuracy.eval({X:mnist.test.images,Y:mnist.test.labels}))

train_nn(X)


