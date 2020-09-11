import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Softmax
from tensorflow.keras.models import Model
import numpy as np
import datetime

class SimplePGNet:

    def __init__(self, state_size, num_actions, learning_rate, vf_coef, ent_coef):

        self.state_size = state_size
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

        input = Input(state_size)

        dense = Dense(64, activation='elu', name='Dense1')(input)

        dense = Dense(32, activation='elu', name='Dense')(dense)
        action_output = Dense(num_actions, activation='softmax', name='Action_Output')(dense)
        value_output = Dense(1, activation=None, name='Value_Output')(dense)

        self.model = Model(inputs=input, outputs=[action_output, value_output])

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def fit_action(self, inputs_mb, target_Q_mb):

        with tf.GradientTape() as tape:
            pred_Q_mb = self.model(inputs_mb)[0]
            neg_log_prob = tf.keras.losses.categorical_crossentropy(y_true=target_Q_mb, y_pred=pred_Q_mb)
            print("T: ", target_Q_mb)
            print("P: ", pred_Q_mb)
            print("T * Log(Q): ", target_Q_mb * tf.math.log(pred_Q_mb))
            print("Sum(T * Log(Q)): ", tf.reduce_sum(-1 * target_Q_mb * tf.math.log(pred_Q_mb), axis=1))
            loss = neg_log_prob
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def fit_value(self, inputs_mb, target_Q_mb):

        with tf.GradientTape() as tape:
            pred_vf_mb = self.model(inputs_mb)[1]
            pred_vf_mb = tf.reshape(pred_vf_mb, target_Q_mb.shape)
            loss = tf.reduce_mean(tf.math.square(target_Q_mb - pred_vf_mb))
            # print("T: ", target_Q_mb)
            # print("P: ", pred_vf_mb)
            # print("Square: ", tf.math.square(target_Q_mb - pred_vf_mb))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def fit_action_value(self, inputs_mb, target_action, target_value):

        with tf.GradientTape() as tape:
            pred_Q_mb, pred_vf_mb = self.model(inputs_mb)
            neg_log_prob = tf.reduce_mean(tf.reduce_sum(-1 * target_action * tf.math.log(pred_Q_mb), axis=1))
            pred_vf_mb = tf.reshape(pred_vf_mb, target_value.shape)
            vf_loss = tf.reduce_mean(tf.square(pred_vf_mb - target_value))
            loss = neg_log_prob + vf_loss

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    @tf.function
    def fit_gradient(self, inputs_mb, target_Q_mb, rewards, values):

        # Mini-batch training
        # advantages = rewards - tf.stop_gradient(values)
        advantages = np.array([1] * rewards.shape[0])

        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)

            pred_Q_mb, pred_vf_mb = self.model(inputs_mb)
            # print("Pred Q: ", pred_Q_mb)
            # print("Target Q: ", target_Q_mb)
            neg_log_prob = tf.reduce_sum(-1. * target_Q_mb * tf.math.log(pred_Q_mb), axis=1)
            # print("NLP: ", neg_log_prob)

            policy_loss = tf.reduce_mean(tf.multiply(neg_log_prob, advantages))
            # print("NLP * Adv: ", tf.multiply(neg_log_prob, advantages))

            entropy_loss = tf.reduce_mean(tf.reduce_sum(-1. * pred_Q_mb * tf.math.log(pred_Q_mb), axis=1))

            pred_vf_mb = tf.reshape(pred_vf_mb, rewards.shape)
            # print("PV: ", pred_vf_mb)
            vf_loss = tf.reduce_mean(tf.square(pred_vf_mb - rewards))
            # print("Square Diff: ", tf.square(pred_vf_mb - rewards))

            loss = policy_loss - self.ent_coef * entropy_loss + self.vf_coef * vf_loss

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return policy_loss, entropy_loss, vf_loss, loss


def test_action_value_fit():
    nbatch = 4
    state_size = (4,)
    num_actions = 2
    X = tf.random.uniform((nbatch, *state_size))

    lr = 0.01
    vf_coef = 1.0
    ent_coef = 0.0

    target_indx = np.random.choice(range(num_actions), size=nbatch)
    target_action_Y = np.zeros((nbatch, num_actions))
    target_action_Y[range(nbatch), target_indx] = 1
    target_action_Y = np.reshape(np.array([[1, 0], [0, 1], [1, 0], [0, 1]]), (nbatch, num_actions))
    target_action_Y = target_action_Y.astype(np.float32)

    target_value_Y = np.array(range(nbatch))
    target_value_Y *= 100
    target_value_Y = target_value_Y.astype(np.float32)


    output_A = []
    output_V = []
    test_loop = 1

    for l in range(test_loop):
        pgnet = SimplePGNet(state_size, num_actions, lr, vf_coef, ent_coef)
        print("Iteration {}...".format(l))
        # print("X: ", X)
        old_action_Y, old_value_Y = pgnet.model(X)
        # print("Old Action Y: ", old_action_Y)
        # print("Old Value Y: ", old_value_Y)
        #

        for _ in range(1000):
            pgnet.fit_gradient(X, target_action_Y, target_value_Y, target_value_Y)

        new_action_Y, new_value_Y = pgnet.model(X)

        a_diff = target_action_Y - new_action_Y
        v_diff = target_value_Y - np.reshape(new_value_Y, target_value_Y.shape)
        # print("A Diff: ", a_diff)
        # print("V Diff: ", v_diff)
        output_A.append(a_diff)
        output_V.append(v_diff)

    output_A = np.array(output_A)
    output_V = np.array(output_V)

    print("A Diff: ", output_A)
    print("V Diff: ", output_V)

    print("A Mean: ", np.mean(output_A, axis=0))
    print("V Mean: ", np.mean(output_V, axis=0))

    print("A MaxErr: ", np.max(np.abs(output_A), axis=0))
    print("V MaxErr: ", np.max(np.abs(output_V), axis=0))

    # for i, layer in enumerate(pgnet.model.layers[1:]):
    #     print("Layer {} W: {}".format(i, layer.get_weights()[0]))
    #     print("Layer {} B: {}".format(i, layer.get_weights()[1]))

def test_value_fit():
    nbatch = 4
    state_size = (4,)
    num_actions = 2
    X = tf.random.uniform((nbatch, *state_size))

    # Random results = learning rate issue?
    lr = 0.1
    vf_coef = 0.5
    ent_coef = 0.0

    output = []
    test_loop = 10

    for l in range(test_loop):
        pgnet = SimplePGNet(state_size, num_actions, lr, vf_coef, ent_coef)

        target_Y = np.array(range(nbatch))

        old_Y = pgnet.model(X)[1]

        for _ in range(1000):
            pgnet.fit_value(X, target_Y)

        print("Update {}:".format(l))
        # print("X: ", X)
        # print("Target Y: ", target_Y)
        # print("Old Y: ", old_Y)
        new_Y = pgnet.model(X)[1]
        # print("New Y: ", new_Y)
        output.append(new_Y)

    output = np.reshape(np.array(output), (test_loop,nbatch))
    print("O: ", output)
    print("O Mean: ", np.mean(output, axis=0))

    # for i, layer in enumerate(pgnet.model.layers[1:]):
    #     print("Layer {} W: {}".format(i, layer.get_weights()[0]))
    #     print("Layer {} B: {}".format(i, layer.get_weights()[1]))

def test_action_fit():
    nbatch = 4
    state_size = (4,)
    num_actions = 2
    X = tf.random.uniform((nbatch, *state_size))

    lr = 0.01
    vf_coef = 0.5
    ent_coef = 0.0

    output = []

    test_loop = 100

    for l in range(test_loop):

        pgnet = SimplePGNet(state_size, 2, lr, vf_coef, ent_coef)

        target_indx = np.random.choice(range(num_actions), size=nbatch)
        target_Y = np.zeros((nbatch, num_actions))
        target_Y[range(nbatch), target_indx] = 1

        old_Y = pgnet.model(X)[0]

        for _ in range(1000):
            pgnet.fit_action(X, target_Y)

        # print("Update {}:".format(l))
        # print("X: ", X)
        # print("Target Y: ", target_Y)
        # print("Old Y: ", old_Y)
        new_Y = pgnet.model(X)[0]
        # print("New Y: ", np.round(new_Y,2))
        diff = target_Y - new_Y
        print(np.round(diff, 6))

        output.append(diff)

    output = np.array(output)
    print("O mean: ", np.mean(output))


if __name__ == "__main__":
    test_action_value_fit()