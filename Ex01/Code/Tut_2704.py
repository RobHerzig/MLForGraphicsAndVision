import tensorflow as tf
import numpy as np
# import helper
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

tf.reset_default_graph()

init = tf.constant([[1.,2.,3.], [4.,5.,6.], [7.,8.,9.]])
A = tf.get_variable("var_A", initializer=init, dtype=tf.float32)

B = tf.get_variable("var_B", initializer=init, dtype=tf.float32)

summ = tf.add(A, B, name='sum')
mul = tf.matmul(A, B, name='mul')

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

x = tf.placeholder("float", None)
y = x * 2

with tf.Session() as session:
    x_data = ([[1, 2, 3], [4, 5, 6], ])
