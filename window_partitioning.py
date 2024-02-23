import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
def window_reverse(windows, window_size, height, width, channels):
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x_win_reverse = tf.reshape(
        windows,
        shape=(-1, patch_num_y, patch_num_x, window_size, window_size, channels),
    )
    x_win_reverse = tf.transpose(x_win_reverse, perm=(0, 1, 3, 2, 4, 5))
    x_win_reverse = tf.reshape(x_win_reverse, shape=(-1, height, width, channels))
    return x_win_reverse

def window_partition(x, window_sz):
    _, h, w, c = x.shape
    patch_num_y = h // window_sz
    patch_num_x = w // window_sz
    x = tf.reshape(
        x, shape=(-1, patch_num_y, window_sz, patch_num_x, window_sz, c)
    )
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = tf.reshape(x, shape=(-1, window_sz, window_sz, c))
    return windows
class EXt_Patch(layers.Layer):
    def __init__(self, pt_size, **kwargs):
        super(EXt_Patch, self).__init__(**kwargs)
        self.patch_size_y = pt_size[0]
        self.patch_size_x = pt_size[0]

    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=(1, self.patch_size_x, self.patch_size_y, 1),
            strides=(1, self.patch_size_x, self.patch_size_y, 1),
            rates=(1, 1, 1, 1),
            padding="VALID",
        )
        patch_dim = patches.shape[-1]
        patch_num = patches.shape[1]
        batch_size = tf.shape(images)[0]
        return tf.reshape(patches, (batch_size, patch_num * patch_num, patch_dim))
class Mer_Patch(tf.keras.layers.Layer):
    def __init__(self, num_patch, embed_dim):
        super(Mer_Patch, self).__init__()
        self.embed_dim = embed_dim
        self.num_patch = num_patch
        self.linear_trans = layers.Dense(2 * embed_dim, use_bias=False)
    def call(self, x):
        height, width = self.num_patch
        _, _, C = x.get_shape().as_list()
        x = tf.reshape(x, shape=(-1, height, width, C))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = tf.concat((x0, x1, x2, x3), axis=-1)
        x = tf.reshape(x, shape=(-1, (height // 2) * (width // 2), 4 * C))
        return self.linear_trans(x)


