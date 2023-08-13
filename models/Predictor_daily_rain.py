import tensorflow as tf

class Predictor(tf.keras.Model):
    """Simple classifier layer to classify the subgroup of data

    Args:
        fc_sizes: Hidden size of the predictor MLP
    """
    def __init__(self, fc_sizes):
        super(Predictor, self).__init__()
        self.fc_sizes = fc_sizes
        self.fc = tf.keras.Sequential([tf.keras.layers.Dense(h, activation=tf.nn.relu, dtype=tf.float32)
                                       for h in fc_sizes])
        self.prob = tf.keras.layers.Dense(1, activation=tf.nn.relu, dtype=tf.float32)

    def __call__(self, local_encs, global_encs, x_lens):
        if not global_encs is None:
            h = tf.concat([local_encs, tf.tile(tf.expand_dims(global_encs, axis=1), [1, local_encs.shape[1], 1])], axis=-1)
            h = tf.keras.layers.BatchNormalization()(h)
        else:
            h = local_encs
        logits = (self.fc(h))
        probs = tf.keras.layers.Dropout(rate=0.3)(self.prob(logits))
        return probs[...,0]