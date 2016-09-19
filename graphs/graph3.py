#encoding: utf-8
'''
Graph representation
'''

import tensorflow as tf


def main_correct():
    '''
    correct: create new graphs, ignore default graph
    '''
    g1 = tf.Graph()
    g2 = tf.Graph()
    with g1.as_default():
        in_graph_g1 = tf.mul(2,3)
        sess_g1 = tf.Session()
        g1_val = sess_g1.run(in_graph_g1)

    with g2.as_default():
        in_graph_g2 = tf.add(2,3)
        sess_g2 = tf.Session()
        g2_val = sess_g2.run(in_graph_g2)

    sess_g1.close()
    sess_g2.close()


def main_correct2():
    '''
    correct: get handle to default graph
    '''
    g1 = tf.get_default_graph()
    g2 = tf.Graph()
    with g1.as_default():
        in_graph_g1 = tf.mul(2,3)
        sess_g1 = tf.Session()
        g1_val = sess_g1.run(in_graph_g1)

    with g2.as_default():
        in_graph_g2 = tf.add(2,3)
        sess_g2 = tf.Session()
        g2_val = sess_g2.run(in_graph_g2)

    sess_g1.close()
    sess_g2.close()


def main_incorrect():
    '''
    incorrect because mix default graph and user-created graph styles
    '''
    g = tf.Graph()

    # in default graph
    in_default_graph = tf.add(1, 2)
    sess = tf.Session()

    with g.as_default():
        # in graph g
        in_graph_g = tf.mul(2, 3)

        sess_g = tf.Session()
        a_val = sess_g.run(in_graph_g)
        print(a_val)

    # in default graph
    also_in_default_graph = tf.sub(5, 1)

    # can get default graph using tf.get_default_graph
    default_graph = tf.get_default_graph()

    sess.run(in_default_graph)
    sess.run(also_in_default_graph)

    writer = tf.train.SummaryWriter('./my_graph', sess.graph)

    writer.close()
    sess_g.close()
    sess.close()

if __name__ == '__main__':
    main_incorrect()
    main_correct()
