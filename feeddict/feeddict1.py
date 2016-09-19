#encoding: utf-8
'''
feeddict1
'''
import tensorflow as tf


def main():
    g = tf.get_default_graph()
    sess = tf.Session(graph=g)
    with g.as_default():
        a = tf.add(2, 5)
        b = tf.mul(a, 3)

        replace_dict = {a: 15}
        # このfeed_dictでbより前のグラフは計算されなくなり、テストやデバッグ時に便利
        b_val = sess.run(b, feed_dict=replace_dict)
        print(b_val)


if __name__ == '__main__':
    main()
