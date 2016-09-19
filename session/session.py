#encoding: utf-8
'''
Session samples
'''
import tensorflow as tf


def main():
    with tf.Session() as sess:
        '''
        sessionのcloseをcontext managerに任せる
        '''
        a = tf.add(2, 3)
        a_val = sess.run(a)
        print(a_val)


def main_default_sess():
    a = tf.mul(3, 6)

    sess = tf.Session()

    with sess.as_default():
        '''
        as defaultのcontext managerを使って、runをeval()で代替できる
        '''
        a_val = a.eval()
        print(a_val)


if __name__ == '__main__':
    main()
    main_default_sess()
