import tensorflow as tf

if __name__=='__main__':
    matrix1 = tf.constant([[3.,3.]])
    matrix2 = tf.constant([[2.,2.]])

    print tf.matmul(m1, m2)