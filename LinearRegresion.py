import tensorflow as tf
import numpy as np
import pandas as pd

###########Data section ##############
X_in = pd.read_pickle('./Data/X.pkl')
Y_out = pd.read_pickle('./Data/Y.pkl')

X = np.reshape(np.array(X_in),[20,1])
Y = np.reshape(np.array(Y_out),[20,1])
# print(X.shape,Y.shape)
#######################################

M = tf.Variable(tf.zeros([1,1]),name='W')
b = tf.Variable(tf.zeros([1]),name='b')

x = tf.placeholder(dtype=tf.float32,shape=[20,1],name='x')

y = tf.matmul(x,M) + b

y_ = tf.placeholder(tf.float32,[20,1],'y_')

cost = tf.reduce_mean(tf.square(y_-y))
cost_hist = tf.summary.histogram('cost',cost)

# optimizer = tf.train.FtrlOptimizer(0.1).minimize(cost) #M: [[9.323375]] b [9.7880125] cost: 140.11472
# optimizer = tf.train.GradientDescentOptimizer(1).minimize(cost)
# optimizer = tf.train.AdamOptimizer(1).minimize(cost) #M: [[9.2943535]] b [10.2214775] cost: 140.0701
# optimizer = tf.train.FtrlOptimizer(1).minimize(cost)#M: [[9.294378]] b [10.220373] cost: 140.07008
optimizer = tf.train.AdagradOptimizer(1).minimize(cost)#M: [[9.294405]] b [10.22001] cost: 140.07011

feed = {x: X, y_: Y}
init = tf.global_variables_initializer()
ind = 0
with tf.Session() as sess:
    # print(Y_out)
    sess.run(init)
    for i in range(10000):
        sess.run(optimizer,feed_dict=feed)
        if i%1500==0:
            print('M:',sess.run(M),'b',sess.run(b),'cost:',sess.run(cost,feed_dict=feed))
    i+=1
    write = tf.summary.FileWriter('./Board', sess.graph)
write.close()














