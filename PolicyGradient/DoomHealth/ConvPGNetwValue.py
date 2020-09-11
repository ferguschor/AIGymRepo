
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, ELU, BatchNormalization, Softmax, Dropout
from tensorflow.keras.models import Model
import numpy as np
import datetime

class ConvPGNet:

    def __init__(self, state_size, num_actions, adv_lr, value_lr):

        input = Input(state_size)

        # Model 2
        conv = Conv2D(filters=16, kernel_size=[8, 8], strides=[4, 4], padding='valid', name='adv_conv1')(input)
        conv = ELU(name='adv_conv1_out')(conv)
        conv = Conv2D(filters=32, kernel_size=[8, 8], strides=[4, 4], padding='valid', name='adv_conv2')(conv)
        conv = ELU(name='adv_conv2_out')(conv)
        conv = Flatten(name="adv_Flatten")(conv)
        fc = Dense(256, activation='elu', name="adv_Dense1")(conv)
        adv_output = Dense(num_actions, activation="softmax", name="Advantage")(fc)

        self.adv_model = Model(inputs=input, outputs=adv_output)

        conv = Conv2D(filters=16, kernel_size=[8, 8], strides=[4, 4], padding='valid', name='value_conv1')(input)
        conv = ELU(name='value_conv1_out')(conv)
        conv = Conv2D(filters=32, kernel_size=[8, 8], strides=[4, 4], padding='valid', name='value_conv2')(conv)
        conv = ELU(name='value_conv2_out')(conv)
        conv = Flatten(name="value_Flatten")(conv)
        fc = Dense(256, activation='elu', name="value_Dense1")(conv)
        value_output = Dense(1, activation=None, name="Value")(fc)

        self.value_model = Model(inputs=input, outputs=value_output)

        self.adv_optimizer = tf.keras.optimizers.Adam(adv_lr)
        self.value_optimizer = tf.keras.optimizers.Adam(value_lr)

        self.loss = tf.keras.losses.categorical_crossentropy


    def fit_gradient(self, inputs_mb, target_Q_mb):

        with tf.GradientTape() as tape:
            pred_Q_mb = self.model(inputs_mb)[0]
            neg_log_prob = tf.keras.losses.categorical_crossentropy(y_true=target_Q_mb, y_pred=pred_Q_mb)
            loss = tf.reduce_mean(neg_log_prob)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss


    def fit_gradient_w_value(self, inputs_mb, target_Q_mb, discounted_episode_rewards):
        gradient_clip = 40
        value_scale = 0.25
        entropy_scaling = 0.05

        with tf.GradientTape() as tape:
            pred_Q_mb = self.adv_model(inputs_mb)
            pred_values = tf.reshape(self.value_model(inputs_mb), discounted_episode_rewards.shape)
            neg_log_prob = tf.keras.losses.categorical_crossentropy(y_true=target_Q_mb, y_pred=pred_Q_mb)
            policy_loss = tf.reduce_mean(tf.multiply(neg_log_prob, discounted_episode_rewards - pred_values))
            entropy_loss = tf.reduce_mean(-entropy_scaling * tf.reduce_sum(pred_Q_mb * tf.math.log(pred_Q_mb), axis=1))
            policy_loss = policy_loss+entropy_loss
        gradients = tape.gradient(policy_loss, self.adv_model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, gradient_clip)
        self.adv_optimizer.apply_gradients(zip(gradients, self.adv_model.trainable_variables))

        with tf.GradientTape() as tape:
            pred_values = tf.reshape(self.value_model(inputs_mb), discounted_episode_rewards.shape)
            value_loss = value_scale * tf.reduce_mean(tf.square(discounted_episode_rewards - pred_values))
        gradients = tape.gradient(value_loss, self.value_model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, gradient_clip)
        self.value_optimizer.apply_gradients(zip(gradients, self.value_model.trainable_variables))

        loss = policy_loss + value_loss

        return policy_loss, value_loss, entropy_loss, loss

if __name__ == "__main__":
    state_size = [84, 84, 4]
    batch_size = 4
    inputs = tf.random.uniform([batch_size, *state_size])
    inputs = tf.reshape(inputs, [-1, *state_size])
    rand_index = np.random.choice(range(3), size=batch_size)
    y_base = np.zeros([batch_size, 3])
    y_base[range(batch_size), rand_index] = 1
    #
    model = ConvPGNet(state_size, num_actions=3, learning_rate=1e-4, entropy_scaling=0.01)
    output = model.model(inputs)
    print(output)
    print(y_base)

    #
    neg_log_prob = tf.keras.losses.categorical_crossentropy(y_true=y_base, y_pred=output[0])
    tmp = y_base * np.log(output[0])
    print(tmp)
    print(neg_log_prob)
    # discounted_rewards = np.array([1, 2, 3, 4])
    # # discounted_rewards = np.reshape(discounted_rewards, output[1].shape)
    # values = tf.reshape(output[1], discounted_rewards.shape)
    # policy_loss = tf.reduce_mean(tf.multiply(neg_log_prob, discounted_rewards - values))
    # entropy_loss = tf.reduce_mean(-0.01 * tf.reduce_sum(output[0] * tf.math.log(output[0]), axis=1))
    # loss = policy_loss + entropy_loss
    # print(loss)
    #
    # loss = model.fit_gradient_w_value(inputs, y_base, discounted_rewards)
    # print(loss)
