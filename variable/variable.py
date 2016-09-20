#encoding: utf-8
'''
Varibles's samples
'''
import tensorflow as tf


def variable_sess():
    sess1 = tf.Session()
    sess2 = tf.Session()

    my_var = tf.Variable(0)

    init = tf.initialize_all_variables()

    sess1.run(init)
    my_var_sess1 = sess1.run(my_var.assign_add(5))
    print("my_var_sess1: %s" % (my_var_sess1))

    sess2.run(init)
    my_var_sess2 = sess2.run(my_var.assign_add(2))
    print("my_var_sess2: %s" % (my_var_sess2))

    sess1.run(init)
    print("init sess1: %s" % (sess1.run(my_var)))

    print("assign sess1: %s" % (sess1.run(my_var.assign(10))))

def variable_change():
    my_var = tf.Variable(1)

    my_var_times_two = my_var.assign(my_var * 2)

    init = tf.initialize_all_variables()
    
    with tf.Session() as sess:
        sess.run(init)
        print("times two: %s" % sess.run(my_var_times_two))
        print("times two 2nd: %s" % sess.run(my_var_times_two))


        # for increment
        my_var_add = sess.run(my_var.assign_add(1))
        print("my_var_add: %s" % (my_var_add))
        my_var_sub = sess.run(my_var.assign_sub(1))
        print("my_var_sub: %s" % (my_var_sub))


def variable_define():
    my_var = tf.Variable(3, name="my_variable")
    add = tf.add(5, my_var)
    mul = tf.mul(8, my_var)

    # 2x2 matrix of zeros
    zeros = tf.zeros([2, 2])

    # vector of length 6 of ones
    ones = tf.ones([6])

    # 3x3x3 Tensor of random uniform values between 0 and 10
    uniform = tf.random_uniform([3, 3, 3], minval=0, maxval=10)

    # 3x3x3 Tensor of nomally distributed numbers; mean0 and standard deviation 2
    normal = tf.random_normal([3, 3, 3], mean=0.0, stddev=2.0)

    # 制約付きガウシアンもつかえる(No values below 3.0 or above 7.0 will be returned in this Tensor)
    trunc = tf.truncated_normal([2, 2], mean=5.0, stddev=1.0) 

    # 上記のガウシアンを初期値としたTensor Variable
    random_var = tf.Variable(tf.truncated_normal([2, 2]))

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        print("add: %s" % sess.run([add]))

    

    

if __name__ == '__main__':
    variable_define()
    variable_change()
    variable_sess()
