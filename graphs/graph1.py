#encoding: utf-8
'''
first simple graphs
'''
import tensorflow as tf


def main():
    '''
    graph1 definition
    '''
    a = tf.constant(5, name="input_a")
    b = tf.constant(3, name="input_b")
    c = tf.mul(a, b, name="mulc_c")
    d = tf.add(a, b, name="add_d")
    e = tf.add(c, d, name="add_e")

    sess = tf.Session()

    writer = tf.train.SummaryWriter('./my_graph', sess.graph)

    e_val = sess.run(e)
    print(e_val)

    writer.close()
    sess.close()

if __name__ == "__main__":
    main()
