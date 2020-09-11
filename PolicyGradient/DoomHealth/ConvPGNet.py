
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, ELU, BatchNormalization, Softmax, Dropout
from tensorflow.keras.models import Model
import numpy as np
import datetime

class ConvPGNet:

    def __init__(self, state_size, num_actions, learning_rate):

        input = Input(shape=state_size)
        # Model Layers
        conv = Conv2D(filters=32, kernel_size=[8,8], strides=[4,4], padding='valid', name='conv1')(input)
        conv = BatchNormalization(epsilon=1e-5, name='batch_norm1')(conv)
        conv = ELU(name='conv1_out')(conv)
        conv = Conv2D(filters=64, kernel_size=[4, 4], strides=[2, 2], padding='valid', name='conv2')(conv)
        conv = BatchNormalization(epsilon=1e-5, name='batch_norm2')(conv)
        conv = ELU(name='conv2_out')(conv)
        conv = Conv2D(filters=128, kernel_size=[4, 4], strides=[2, 2], padding='valid', name='conv3')(conv)
        conv = BatchNormalization(epsilon=1e-5, name='batch_norm3')(conv)
        conv = ELU(name='conv3_out')(conv)
        conv = Flatten(name="Flatten")(conv)
        fc = Dense(512, activation='elu', name="Dense1")(conv)
        output = Dense(num_actions, activation="softmax", name="Dense2")(fc)

        self.model = Model(inputs=input, outputs=output)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def fit_gradient(self, inputs_mb, target_Q_mb, discounted_episode_rewards, tf_writer, step):
        entropy_scaling = 0.01

        with tf.GradientTape() as tape:
            pred_Q_mb = self.model(inputs_mb)
            neg_log_prob = -1 * tf.reduce_sum(target_Q_mb * tf.math.log(pred_Q_mb), axis=1)
            # neg_log_prob = tf.keras.losses.categorical_crossentropy(y_true=target_Q_mb, y_pred=pred_Q_mb)
            # entropy_loss = tf.reduce_mean(-entropy_scaling * tf.reduce_sum(pred_Q_mb * tf.math.log(pred_Q_mb), axis=1))
            policy_loss = tf.reduce_mean(tf.multiply(neg_log_prob, discounted_episode_rewards))
            loss = policy_loss
        gradients = tape.gradient(loss, self.model.trainable_variables)

        print("lr at iteration {}: {}".format(self.optimizer.iterations + 1, self.optimizer._decayed_lr('float32').numpy()))

        with tf_writer.as_default():
            for i, grads in enumerate(gradients):
                tf.summary.histogram("Layer"+str(i)+"Grads", grads, step=step)
                tf.summary.scalar("Layer"+str(i)+"GradsMean", np.mean(grads), step=step)
                tf.summary.scalar("Layer"+str(i)+"GradsStd", np.std(grads), step=step)

        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

if __name__ == "__main__":
    state_size = [84, 84, 4]
    batch_size = 4
    inputs = tf.random.uniform([batch_size, *state_size])
    inputs = tf.reshape(inputs, [-1, *state_size])
    rand_index = np.random.choice(range(3), size=batch_size)
    y_base = np.zeros([batch_size, 3])
    y_base[range(batch_size), rand_index] = 1

    model = ConvPGNet(state_size, num_actions=3, learning_rate=1e-4)
    output = model.model(inputs)

    discounted_rewards = np.array([1 for i in range(batch_size)])

    model.fit_gradient(inputs, y_base, discounted_rewards)
