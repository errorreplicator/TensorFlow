import tensorflow as tf

a = tf.constant(5.6, name='a')
b = tf.constant(4.0, name='b')
c = tf.constant(9.0,name='c')
d = tf.constant(3.0, name='d')

sum = tf.add(a,b,name='add')
minus = tf.subtract(c,b, name='minus')
squer = tf.sqrt(c, name='sqer')

sum_all = tf.add_n([sum,minus,squer],name='sum_all')

with tf.Session() as sess:
    sess.run(sum_all)
    print('Sum: ',sess.run(sum))
    print('Dif:',sess.run(minus))
    print('Sqer: ', sess.run(squer))
    print('Suma wszystkich sum:',sess.run(sum_all))
    nowa_suma = tf.add_n([sum,minus,squer,sum_all],name='nowa_suma')
    print('Nowa Suma: ',sess.run(nowa_suma))
    # print()

    write = tf.summary.FileWriter('./Board',sess.graph)
write.close()

# C:\PythonProj\TensorFlow
# λ tensorboard --logdir=Board --port 6006