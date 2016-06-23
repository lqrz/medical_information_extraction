import tensorflow as tf

import numpy as np

def matrix_multiplication():
    m1 = tf.constant([[3.,3.]])
    m2 = tf.constant([[2.,2.]])

    return tf.matmul(m1, tf.transpose(m2))


def run_matrix_multiplication():
    ses = tf.Session()
    print ses.run(matrix_multiplication())

if __name__=='__main__':
    run_matrix_multiplication()
