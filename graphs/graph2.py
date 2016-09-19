#encoding: utf-8
'''
first simple graphs
'''
import tensorflow as tf


def main():
    '''
    graph2 definition
    '''
    a = tf.constant([5, 3], name="input_a")
    b = tf.reduce_prod(a, name="prod_b")
    c = tf.reduce_sum(a, name="sum_c")
    d = tf.add(b, c, name="add_d")

    sess = tf.Session()

    writer = tf.train.SummaryWriter('./my_graph', sess.graph)

    e_val = sess.run(d)
    print(e_val)

    writer.close()
    sess.close()

if __name__ == "__main__":
    main()
