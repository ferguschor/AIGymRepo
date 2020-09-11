# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from collections import deque
import random

import datetime
import numpy as np

print(tf.__version__)


class DQNetwork:

    def __init__(self, state_size, num_actions, learning_rate, name='DQNetwork'):

        # State_size should be (B, H, W, in_channels)
        self.state_size = state_size
        # left, right, shoot
        self.num_actions = num_actions
        self.learning_rate = learning_rate


        # Model structure
        self.model = keras.models.Sequential([
            keras.layers.Conv2D(input_shape=state_size, filters=32, kernel_size=[8,8], strides=[4,4], padding='valid', name='conv1'),
            keras.layers.BatchNormalization(epsilon=1e-5, name='batch_norm1'),
            keras.layers.ELU(name='conv1_out'),
            keras.layers.Conv2D(filters=64, kernel_size=[4, 4], strides=[2, 2], padding='valid',
                                name='conv2'),
            keras.layers.BatchNormalization(epsilon=1e-5, name='batch_norm2'),
            keras.layers.ELU(name='conv2_out'),
            keras.layers.Conv2D(filters=128, kernel_size=[4, 4], strides=[2, 2], padding='valid',
                                name='conv3'),
            keras.layers.BatchNormalization(epsilon=1e-5, name='batch_norm3'),
            keras.layers.ELU(name='conv3_out'),
            keras.layers.Flatten(name="Flatten"),
            keras.layers.Dense(512, activation='elu', name="Dense1"),
            keras.layers.Dense(num_actions, activation=None,  name="Dense2")

            # keras.layers.Flatten(input_shape=state_size, name="Flatten"),
            # keras.layers.Dense(512, activation='elu', name="Dense1"),
            # keras.layers.Dense(3, activation=None, name="Dense2")
        ])

        # Loss should be mean squared error of y_pred vs y_target
        self.loss = keras.losses.mean_squared_error
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Create the model
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss)

        self.train_loss = keras.metrics.Mean("train_loss", dtype=tf.float32)


    def single_action_train(self, state_mb, action_mb, target_Qs_mb):
        # Loss = Squared difference between the Q for the selected action and the target Q
        # Calculates the loss for each sample
        with tf.GradientTape() as tape:
            output_mb = self.model(state_mb)
            current_Qs_mb = tf.reduce_sum(tf.multiply(output_mb, action_mb), axis=1)
            loss = keras.losses.mean_squared_error(y_true=target_Qs_mb, y_pred=current_Qs_mb)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)

    def fit_Gradient(self, state_mb, target_Qs_mb):
        # Loss = Squared difference between the Q for the selected action and the target Q
        # Calculates the loss for each sample
        with tf.GradientTape() as tape:
            current_Qs_mb = self.model(state_mb)
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(target_Qs_mb-current_Qs_mb), axis=1))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # for i, item in enumerate(gradients):
        #     print("Gradient{}: {}".format(i,item.shape))
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)

# Memory used to keep track of states
class Memory():
    def __init__(self, memory_size):
        self.buffer = deque(maxlen=memory_size)


    def append(self, state):
        self.buffer.append(state)

    def sample(self, batch_size):
        memory_size = len(self.buffer)
        rand_indices = np.random.choice(memory_size, size=batch_size, replace=True)
        memory_sample = [self.buffer[i] for i in rand_indices]
        return memory_sample

    def __len__(self):
        return len(self.buffer)

if __name__=="__main__":
    state_size = [84, 84, 4]
    # num_actions = game.get_available_buttons_size()
    num_actions = 3
    learning_rate = 0.0002


    DQN = DQNetwork (state_size=state_size, num_actions=1, learning_rate=learning_rate)
    # print(DQN.model.layers)
    # test = tf.Variable([-2, 1, 2, 3, -1, -4, -1311])
    # o = tf.nn.relu(test)
    # print (test)
    # print (o)
    #
    # w = DQN.model.layers[0].get_weights()[0]
    # b = DQN.model.layers[0].get_weights()[1]
    # print(w.shape)
    # print(b.shape)
    img = tf.random.uniform((10, 84, 84, 4))
    print(img)
    img = tf.reshape(img, (-1,84,84,4))
    output = DQN.model(img)
    print(type(output))
    print(output.shape)
    # print(output[:,1])
    # print(np.argmax(output, axis=1))
    # print(np.max(output, axis=1))
    # print(np.max(output, axis=0))


