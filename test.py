import tensorflow as tf

BATCH_SIZE = 1000

@tf.function
def test():
    x = 0

    for i in range(100000):
        y = last_row = tf.tile([[[0.0, 0.0, 1.0]]], [BATCH_SIZE, 1, 1])
        x += 1

    return x

test()
