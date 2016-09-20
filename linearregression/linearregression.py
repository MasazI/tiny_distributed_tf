# encoding: utf-8
'''
linear regression sample using TensorFlow
'''
import tensorflow as tf


def inference(x):
    w = tf.Variable(tf.zeros([2, 1]), name="weights")
    b = tf.Variable(0., name="bias")

    return tf.matmul(x, w) + b


def get_loss(x, y):
    return tf.reduce_sum(tf.squared_difference(x, y))


def train():
    input, target = inputs()

    logits = inference(input)

    loss = get_loss(logits, target)

    train_ops = get_train_ops(loss)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        training_steps = 1000
        for step in xrange(training_steps):
            sess.run([train_ops])

            if step % 10 == 0:
                predicts, loss_eval = sess.run([logits, loss])
                print("loss: %s, %s" % (predicts, loss_eval))

        coord.request_stop()
        coord.join(threads)
        sess.close()


def get_train_ops(loss):
    learning_rate = 0.0000001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
 

def inputs():
    weight_age = [[84, 46], [73, 20], [65, 52], [70, 30], [76, 57], [69, 25],
        [63, 28], [72, 36], [79, 57], [75, 44], [27, 24], [89, 31], [65, 52],
        [57, 23], [59, 60], [69, 48], [60, 34], [79, 51], [75, 50], [82, 34], 
        [59, 46], [67, 23], [85, 37], [55, 40], [63, 30]]
    blood_fat_content = [354, 190, 405, 263, 451, 302, 288, 385, 402, 365, 209, 
        290, 346, 254, 395, 434, 220, 374, 308, 220, 311, 181, 274, 303, 244]
    return tf.to_float(weight_age), tf.to_float(blood_fat_content)


if __name__ == '__main__':
    train()
