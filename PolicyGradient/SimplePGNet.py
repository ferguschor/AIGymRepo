
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np
import datetime

class SimplePGNet:

    def __init__(self, state_size, num_actions, learning_rate):

        input = Input(state_size)


        dense = Dense(10, activation='relu', name='Dense1')(input)
        dense = Dense(num_actions, activation='relu', name='Dense2')(dense)
        output = Dense(num_actions, activation='softmax', name='Output')(dense)

        self.model = Model(inputs=input, outputs=output)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def fit_gradient(self, inputs_mb, target_Q_mb, discounted_episode_rewards):

        with tf.GradientTape() as tape:
            pred_Q_mb = self.model(inputs_mb)
            neg_log_prob = tf.keras.losses.categorical_crossentropy(y_true=target_Q_mb, y_pred=pred_Q_mb)
            loss = tf.reduce_mean(tf.multiply(neg_log_prob, discounted_episode_rewards))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    def fit_gradient_test(self, inputs_mb, target_Q_mb):

        with tf.GradientTape() as tape:
            pred_Q_mb = self.model(inputs_mb)
            neg_log_prob = tf.keras.losses.binary_crossentropy(y_true=target_Q_mb, y_pred=pred_Q_mb)
            loss = neg_log_prob
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss


if __name__ == "__main__":

    # Initializiation issue?

    nbatch = 4
    state_size = (4,)
    num_actions = 2
    X = tf.random.uniform((nbatch, *state_size))

    lr = 0.01
    vf_coef = 0.5
    ent_coef = 0.0
    pgnet = SimplePGNet(state_size, 2, lr)

    target_indx = np.random.choice(range(num_actions), size=nbatch)
    target_Y = np.zeros((nbatch, num_actions))
    target_Y[range(nbatch), target_indx] = 1
    target_Y = np.reshape(np.array([[1,0], [0,1], [0,1], [1,0]]), (4,2))

    old_Y = pgnet.model(X)

    for _ in range(100):
        pgnet.fit_gradient_test(X, target_Y)

    print("X: ", X)
    print("Target Y: ", target_Y)
    print("Old Y: ", old_Y)
    print("New Y: ", np.round(pgnet.model(X), 2))