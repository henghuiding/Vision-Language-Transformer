import keras.backend as K
import tensorflow as tf


def expand_and_tile(x, outsize):
    x = K.expand_dims(x, axis=1)
    x = K.expand_dims(x, axis=1)
    x = K.tile(x, [1, outsize, outsize, 1])
    return x


def expand_and_tile_1(x, outchannels):
    x = K.expand_dims(x, axis=-1)
    x = K.tile(x, [1, 1, outchannels])
    return x


def normalize_by_dim(x, dim=1024.):
    d = tf.convert_to_tensor(dim)
    return x/K.sqrt(d)


def split_dim_concat_batch(x, n):
    return tf.concat(tf.split(x, n, axis=-1), axis=0)


def split_batch_concat_dim(x, n):
    return tf.concat(tf.split(x, n, axis=0), axis=-1)


def normalize(x):
    x = (x+1.)/2.
    return K.clip(x, 1e-6, 1.-1e-6)


def l2_normalize(x):
    return tf.nn.l2_normalize(x, axis=-1, epsilon=1e-6)


def softmax(x):
    return K.softmax(x-tf.reduce_max(x), -1)


def concat_coord(x):
    ins_feat = x  # [N, h, w, c]

    batch_size = tf.shape(x)[0]
    h = tf.shape(x)[1]
    w = tf.shape(x)[2]
    float_h = K.cast(h, 'float32')
    float_w = K.cast(w, 'float32')

    y_range = K.arange(float_h, dtype='float32')     # [h, ]
    y_range = 2.0 * y_range / (float_h - 1.0) - 1.0
    x_range = K.arange(float_w, dtype='float32')     # [w, ]
    x_range = 2.0 * x_range / (float_w - 1.0) - 1.0
    x_range = x_range[None, :]   # [1, w]
    y_range = y_range[:, None]   # [h, 1]
    x = K.tile(x_range, [h, 1])     # [h, w]
    y = K.tile(y_range, [1, w])     # [h, w]

    x = x[None, :, :, None]   # [1, h, w, 1]
    y = y[None, :, :, None]   # [1, h, w, 1]
    x = K.tile(x, [batch_size, 1, 1, 1])   # [N, h, w, 1]
    y = K.tile(y, [batch_size, 1, 1, 1])   # [N, h, w, 1]

    ins_feat_out = K.concatenate([ins_feat, x, x, x, y, y, y])   # [N, h, w, c+6]

    return ins_feat_out
