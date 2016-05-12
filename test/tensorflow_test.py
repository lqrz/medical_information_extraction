import tensorflow as tf

def matrix_multiplication():
    m1 = tf.constant([[3.,3.]])
    m2 = tf.constant([[2.,2.]])

    return tf.matmul(m1, tf.transpose(m2))

if __name__=='__main__':
    ses = tf.Session()
    print ses.run(matrix_multiplication())