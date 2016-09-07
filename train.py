#encoding: utf-8

from datetime import datetime
import os.path
import time

import tensorflow.python.platform
from tensorflow.python.platform import gfile

#from PIL import Image

import numpy as np
from six.moves import xrange
import tensorflow as tf

# model
import model as model

# train operation
import train_op as op

# inputs
import dataset

# distributed
# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

DFLAGS = tf.app.flags.FLAGS

# settings
import settings
FLAGS = settings.FLAGS

TRAIN_DIR = FLAGS.train_dir
MAX_STEPS = FLAGS.max_steps
LOG_DEVICE_PLACEMENT = FLAGS.log_device_placement
TF_RECORDS = FLAGS.train_tfrecords
BATCH_SIZE = FLAGS.batch_size

from datetime import datetime as dt
tdatetime = dt.now()
train_start_time = tdatetime.strftime('%Y%m%d%H%M%S')


def main(_):
    if gfile.Exists(TRAIN_DIR):
        gfile.DeleteRecursively(TRAIN_DIR)
    gfile.MakeDirs(TRAIN_DIR)

    # locally
    #train()

    print("ps: %s" % (DFLAGS.task_index))

    ps_hosts = DFLAGS.ps_hosts.split(",")
    worker_hosts = DFLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if DFLAGS.job_name == "ps":
        server.join()
    elif DFLAGS.job_name == "worker":
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % DFLAGS.task_index,
                cluster=cluster)):
            # step num of global
            global_step = tf.Variable(0, trainable=False)

            # training data
            filename_queue = tf.train.string_input_producer(["output/data/airquality.csv"])
            datas, targets = dataset.mini_batch(filename_queue, BATCH_SIZE)

            # inference
            logits = model.inference(datas)

            debug_value = model.debug(logits)

            # loss graphのoutputとlabelを利用
            loss = model.loss(logits, targets)

            global_step = tf.Variable(0)

            #train_op = tf.train.AdagradOptimizer(0.0001).minimize(
            #    loss, global_step=global_step)
            train_op = op.train(loss, global_step)

            saver = tf.train.Saver()
            summary_op = tf.merge_all_summaries()
            init_op = tf.initialize_all_variables()

        # Create a "supervisor", which oversees the training process.
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir="/tmp/train_logs",
                                 init_op=init_op,
                                 init_feed_dict=None,
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=60)

        # The supervisor takes care of session initialization, restoring from
        # a checkpoint, and closing when done or an error occurs.
        with sv.managed_session(server.target) as sess:
            # Loop until the supervisor shuts down or 1000000 steps have completed.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            step = 0
            while not sv.should_stop() and step < 1000000:
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.
                start_time = time.time()
                _, loss_value, predict_value, targets_eval, step = sess.run([train_op, loss, debug_value, targets, global_step])
                #_, step = sess.run([train_op, global_step])
                duration = time.time() - start_time

                if step % 100 == 0:
                    # mini batch size
                    num_examples_per_step = BATCH_SIZE

                    # examples num per sec
                    examples_per_sec = num_examples_per_step / duration

                    # duration per batch
                    sec_per_batch = float(duration)

                    # time, step num, loss, exampeles num per sec, time per batch
                    format_str = '$s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)'
                    print str(datetime.now()) + ': step' + str(step) + ', loss= ' + str(loss_value) + ' ' + str(
                        examples_per_sec) + ' examples/sec; ' + str(sec_per_batch) + ' sec/batch'
                    print "predict: ", predict_value
                    print "targets: ", targets_eval

            coord.request_stop()
            coord.join(threads)
            sess.close()

        # Ask for all the services to stop.
        sv.stop()


if __name__ == "__main__":
    tf.app.run()

