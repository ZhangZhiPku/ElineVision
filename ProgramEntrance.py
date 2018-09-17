# ElineVision is designed to help optimize a neural network with visual method.

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as mpl
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

_idx = 0
_x_dimension = 2
_plot_resolution = 25
_extension = 2
np.random.seed(0)

def prepare_data(samples=1000, x_dimension=_x_dimension):

    _shape_of_x = [samples, x_dimension]
    _shape_of_y = [samples, 1]

    x = np.random.normal(loc=0, scale=3, size=_shape_of_x)
    y = np.sum(x, axis=1)
    y = y + np.random.normal(loc=0, scale=3, size=[len(y)])

    return x, y


def build_network(input_batch_shape=(_x_dimension, ), n_layers=4):
    model = tf.keras.Sequential()

    input_layer = tf.keras.layers.Dense(units=32, activation='linear',
                                        input_shape=input_batch_shape,
                                        use_bias=False)
    output_layer = tf.keras.layers.Dense(units=1, activation='linear')

    input_layer.trainable=False
    output_layer.trainable=False

    _ = np.array(input_layer.get_weights()).shape
    input_layer.set_weights(np.ones(_))

    _ = np.array(output_layer.get_weights()).shape
    output_layer.set_weights(np.ones(_))

    layers = [tf.keras.layers.Dense(units=32, activation='tanh', use_bias=False) for i in range(n_layers)]

    model.add(input_layer)
    for _layer in layers:
        model.add(_layer)
    model.add(output_layer)

    model.compile(
        optimizer=tf.keras.optimizers.SGD(lr=1e-3),
        loss=tf.keras.losses.mean_squared_error,
        metrics=[tf.keras.metrics.mean_squared_error]
    )

    return model, layers


def plot_surface(plot_df, resolution=25):
    fig = mpl.figure()
    ax = Axes3D(fig)

    x, y, z = plot_df['X'], plot_df['Y'], plot_df['Z'] * 0.01

    x = np.array(x).reshape([resolution, resolution])
    y = np.array(y).reshape([resolution, resolution])
    z = np.array(z).reshape([resolution, resolution])

    ax.plot_surface(x, y, z, cmap=mpl.cm.coolwarm, rstride=1, cstride=1)

    ax.set_xlabel("x-label", color='r')
    ax.set_ylabel("y-label", color='g')
    ax.set_zlabel("z-label", color='b')

    mpl.show()


if __name__ == '__main__':
    x, y = prepare_data(samples=128)

    _target_epochs = 12800
    _batch_size = 128
    _verbose_interval = 128

    _model, _layers = build_network()

    _rxs, _rys = [], []
    for _layer in _layers:
        _weights_shape = np.array(_layer.get_weights()).shape
        _rxs.append(np.random.normal(size=_weights_shape))
        _rys.append(np.random.normal(size=_weights_shape))

    _multipliers = [i / _plot_resolution / 10 for i in range(_plot_resolution + 1)]
    _losses = []

    for _x_multiplier in _multipliers:
        for _y_multiplier in _multipliers:
            for _rx, _ry, _layer in zip(_rxs, _rys, _layers):
                _layer.set_weights(np.add(_rx * _x_multiplier, _ry * _y_multiplier))
            _losses.append((_x_multiplier, _y_multiplier, _model.evaluate(x, y)[0]))

    df = pd.DataFrame(data=_losses, columns=['X', 'Y', 'Z'])
    plot_surface(df, resolution=_plot_resolution + 1)

    #_model.fit(x, y, batch_size=_batch_size, epochs=_target_epochs)
