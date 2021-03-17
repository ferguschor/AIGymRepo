# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, ELU, BatchNormalization, Dense, Flatten
from collections import deque
import random

import time
import datetime
import numpy as np

print(tf.__version__)


class DuelDQNetwork:

    def __init__(self, state_size, num_actions, learning_rate, name='DQNetwork'):

        # State_size should be (B, H, W, in_channels)
        self.state_size = state_size
        # left, right, shoot
        self.num_actions = num_actions
        self.learning_rate = learning_rate

        # Model structure
        # Input transforms to tensor
        input = Input(shape=state_size)

        # Model Layers
        conv = Conv2D(filters=32, kernel_size=[8,8], strides=[4,4], padding='valid', name='conv1')(input)
        # conv = BatchNormalization(epsilon=1e-5, name='batch_norm1')(conv)
        conv = ELU(name='conv1_out')(conv)
        conv = Conv2D(filters=64, kernel_size=[4, 4], strides=[2, 2], padding='valid', name='conv2')(conv)
        # conv = BatchNormalization(epsilon=1e-5, name='batch_norm2')(conv)
        conv = ELU(name='conv2_out')(conv)
        conv = Conv2D(filters=128, kernel_size=[4, 4], strides=[2, 2], padding='valid', name='conv3')(conv)
        # conv = BatchNormalization(epsilon=1e-5, name='batch_norm3')(conv)
        conv = ELU(name='conv3_out')(conv)
        conv = Flatten(name="Flatten")(conv)

        # Split streams for dueling networks
        # Network to determine value of states
        state_fc = Dense(512, activation='elu', name="State_FC1")(conv)
        state_fc = Dense(1, activation=None, name="State_Output")(state_fc)
        # Network to determine value of actions
        action_fc = Dense(512, activation='elu', name="Action_FC1")(conv)
        action_fc = Dense(num_actions, activation=None, name="Action_Output")(action_fc)

        output = state_fc + (action_fc - tf.reduce_mean(action_fc, axis=1, keepdims=True))

        self.model = keras.models.Model(inputs=input, outputs=output)

        # Loss should be mean squared error of y_pred vs y_target
        self.loss = keras.losses.mean_squared_error
        self.optimizer = keras.optimizers.RMSprop(learning_rate=self.learning_rate)

        # Create the model
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss)

    def single_action_gradient(self, state_mb, action_mb, target_Qs_mb, ISWeights_mb):
        # Loss = Squared difference between the Q for the selected action and the target Q
        # Calculates the loss for each sample
        with tf.GradientTape() as tape:
            current_Qs_mb = self.model(state_mb)
            current_Qs_mb = tf.reduce_sum(current_Qs_mb * action_mb, axis=1)
            # Weighted product using ISWeights
            loss = tf.reduce_mean(tf.multiply(tf.square(target_Qs_mb - current_Qs_mb), ISWeights_mb))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # absolute errors used to update SumTree
        abs_errors = tf.abs(target_Qs_mb - current_Qs_mb)
        return loss, abs_errors

    def fit_gradient(self, state_mb, target_Qs_mb):
        # Loss = Squared difference between the Q for the selected action and the target Q
        # Calculates the loss for each sample
        with tf.GradientTape() as tape:
            current_Qs_mb = self.model(state_mb)
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(target_Qs_mb-current_Qs_mb), axis=1))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # for i, item in enumerate(gradients):
        #     print("Gradient{}: {}".format(i,item.shape))
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


    def weighted_fit_gradient(self, state_mb, target_Qs_mb, ISWeights_mb):
        # Loss = Squared difference between the Q for the selected action and the target Q
        # Calculates the loss for each sample
        with tf.GradientTape() as tape:
            current_Qs_mb = self.model(state_mb)
            # print("Current_Q_mb: \n{}".format(current_Qs_mb))
            # Weighted product using ISWeights
            loss = tf.reduce_mean(tf.multiply(tf.reduce_sum(tf.square(target_Qs_mb - current_Qs_mb), axis=1), ISWeights_mb))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # absolute errors used to update SumTree
        abs_errors = tf.reduce_sum(tf.abs(target_Qs_mb - current_Qs_mb), axis=1)
        return loss, abs_errors


if __name__=="__main__":
    start = time.time()

    state_size = [84, 84, 4]
    # num_actions = game.get_available_buttons_size()
    num_actions = 3
    learning_rate = 0.0002


    DQN = DuelDQNetwork (state_size=state_size, num_actions=num_actions, learning_rate=learning_rate)
    target = DuelDQNetwork (state_size=state_size, num_actions=num_actions, learning_rate=learning_rate)


    tf.random.set_seed(42)
    input = tf.random.uniform((4, 84, 84, 4))
    # input = tf.reshape(input, (-1,84,84,4))
    zeroes = tf.zeros((1, *state_size))
    input = tf.concat((input, zeroes), axis=0)

    y_pred = DQN.model(input)
    print("Before training: \n", y_pred)

    y_true = np.array([[50, 40, 1], [1, 33, 95], [0, 1, 200], [1, 22, 0], [-100, 0, 0]])

    for i in range(1000):
        DQN.fit_gradient(input, y_true)

    y_pred = DQN.model(input)
    print("After training: \n", y_pred)


    # y_true = DQN.model(input)
    # y_true = np.array(y_true)
    # y_true[range(y_true.shape[0]), [0, 1, 2, 2]] = [1, 2, 3, 4]
    # print(y_true)
    #
    # ISWeights_mb = [0.4, 0.3, 0.1, 0.2]
    #
    #
    # with tf.GradientTape() as tape:
    #     y_pred = DQN.model(input)
    #     print(y_pred)
    #     loss = tf.multiply(tf.reduce_sum(tf.square(y_pred - y_true), axis=1), ISWeights_mb)
    # grad = tape.gradient(loss, DQN.model.trainable_variables)
    # # for i in grad:
    # #     print(i.shape)
    # print(y_pred - y_true)
    # print(tf.square(y_pred - y_true))
    # print(tf.reduce_sum(tf.square(y_pred - y_true), axis=1))
    # print(loss)
    #
    # abs_errors = tf.reduce_sum(tf.abs(y_true - y_pred), axis=1)
    # print(abs_errors)


