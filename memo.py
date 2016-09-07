#encoding: utf-8
'''
tiny cluster example
'''
import tensorflow as tf

c = tf.constant("distributed_tf")

# Serverの生成
server = tf.train.Server.create_local_server()

# Server上のセッション
sess = tf.Session(server.target)

message = sess.run(c)

print(message)