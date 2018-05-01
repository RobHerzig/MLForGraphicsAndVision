#NAME: ROBERT HERZIG
#MATNO: 3605172
#COMPILED WITH PYTHON 3.6.5
import tensorflow as tf
import numpy as np

pi_val = np.pi
msg_op = tf.constant('Pi is approximately %f' %pi_val)

with tf.Session() as sess:
    sess.run(msg_op)
    print(msg_op.eval().decode())
