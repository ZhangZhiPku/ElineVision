# ElineVision is designed to help optimize a neural network with visual method.

import numpy as np
import tensorflow as tf

_idx = 0


def prepare_data(samples=1000, x_dimension=10):

    _shape_of_x = [samples, x_dimension]
    _shape_of_y = [samples, 1]

    x = np.random.normal(loc=0, scale=0.5, size=_shape_of_x)
    y = np.random.normal(loc=0, scale=0.5, size=_shape_of_y)

    return x, y


def build_network(input_batch_shape=(8, 32)):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=input_batch_shape))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Dense(units=1, activation='relu'))

    model.compile(
        optimizer=tf.keras.optimizers.SGD(lr=1e-3, momentum=0.1),
        loss=tf.keras.losses.mean_squared_error,
        metrics=[tf.keras.metrics.mean_squared_error]
    )

    return model


def make_batch(x, y, batch_size=128):
    global _idx

    if batch_size > len(x):
        raise Exception('can not assign a batch size much larger than samples count.')

    if _idx + batch_size < len(x):
        return x[_idx: _idx + batch_size], y[_idx: _idx + batch_size]
    else:
        return x[_idx: ] + x[: len(x) - _idx], y[_idx: ] + y[: len(x) - _idx]

if __name__ == '__main__':
    x, y = prepare_data(samples=25600, x_dimension=10)

    _target_epochs = 12800
    _batch_size = 128
    _verbose_interval = 128

    _model = build_network()

    for i in range(_target_epochs):
        _x, _y = make_batch(x, y, batch_size=_batch_size)
        _model.fit(_x, _y)

        if i % _verbose_interval is 0:
            print('model fit process at %d, with loss %.2f' % (i, _model.loss))