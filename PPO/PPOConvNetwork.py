import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, ELU, BatchNormalization, Flatten
from tensorflow.keras.models import Model
import numpy as np

class PGNetwork:

    def __init__(self, input_shape, num_actions,  lr, vf_coef, ent_coef):

        self.state_size = input_shape
        self.num_actions = num_actions
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

        input = Input(shape=input_shape)

        conv = Conv2D(filters=32, kernel_size=[8, 8], strides=[4, 4], name="Conv1")(input)
        # conv = BatchNormalization(epsilon=1e-5, name="Conv1_BN")(conv)
        conv = ELU(name="Conv1_ELU")(conv)
        conv = Conv2D(filters=64, kernel_size=[4, 4], strides=[2, 2], name="Conv2")(conv)
        # conv = BatchNormalization(epsilon=1e-5, name="Conv2_BN")(conv)
        conv = ELU(name="Conv2_ELU")(conv)
        conv = Conv2D(filters=128, kernel_size=[4, 4], strides=[2, 2], name="Conv3")(conv)
        # conv = BatchNormalization(epsilon=1e-5, name="Conv3_BN")(conv)
        conv = ELU(name="Conv3_ELU")(conv)

        flatten = Flatten(name="Flatten")(conv)
        dense = Dense(512, activation="elu", name="Dense1")(flatten)
        actions_output = Dense(num_actions, activation="softmax", name="Actions_Output")(dense)
        val_output = Dense(1, activation=None, name="Value_Output")(dense)

        self.model = Model(inputs=input, outputs=[actions_output, val_output])

        self.optimizer = tf.keras.optimizers.Adam(lr, decay=0.99)


    def fit_gradient(self, inputs_mb, target_Q_mb, rewards, values):

        advantages = rewards - tf.stop_gradient(values)
        # advantages = np.array([1] * rewards.shape[0])  # used to test function

        with tf.GradientTape() as tape:

            pred_Q_mb, pred_vf_mb = self.model(inputs_mb)
            # print("Pred Q: ", pred_Q_mb)
            # print("Target Q: ", target_Q_mb)
            neg_log_prob = tf.reduce_sum(-1 * target_Q_mb * tf.math.log(pred_Q_mb), axis=1)
            # print("NLP: ", neg_log_prob)

            policy_loss = tf.reduce_mean(tf.multiply(neg_log_prob, advantages))
            # print("NLP * Adv: ", tf.multiply(neg_log_prob, advantages))

            entropy_loss = tf.reduce_mean(tf.reduce_sum(-1 * pred_Q_mb * tf.math.log(pred_Q_mb), axis=1))

            pred_vf_mb = tf.reshape(pred_vf_mb, rewards.shape)
            # print("PV: ", pred_vf_mb)
            vf_loss = tf.reduce_mean(tf.square(pred_vf_mb - rewards))
            # print("Square Diff: ", tf.square(pred_vf_mb - rewards))

            loss = policy_loss - self.ent_coef * entropy_loss + self.vf_coef * vf_loss
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return policy_loss, entropy_loss, vf_loss, loss


def test_action_value_fit():
    nbatch = 4
    state_size = (96, 96, 4)
    num_actions = 7
    X = tf.random.uniform((nbatch, *state_size))

    lr = 0.001
    vf_coef = 0.5
    ent_coef = 0.0

    target_indx = np.random.choice(range(num_actions), size=nbatch)
    target_action_Y = np.zeros((nbatch, num_actions))
    target_action_Y[range(nbatch), target_indx] = 1.
    # target_action_Y = np.reshape(np.array([[1, 0], [0, 1], [1, 0], [0, 1]]), (nbatch, num_actions))

    target_value_Y = np.array(range(nbatch))

    output_A = []
    output_V = []
    test_loop = 1

    for l in range(test_loop):
        pgnet = PGNetwork(state_size, num_actions, lr, vf_coef, ent_coef)
        print("Iteration {}...".format(l))
        # print("X: ", X)
        old_action_Y, old_value_Y = pgnet.model(X)
        # print("Old Action Y: ", old_action_Y)
        # print("Old Value Y: ", old_value_Y)
        #
        # print("Target Action Y: ", target_action_Y)
        # print("Target Value Y: ", target_value_Y)

        for _ in range(1000):
            pgnet.fit_gradient(X, target_action_Y, target_value_Y, target_value_Y)

        new_action_Y, new_value_Y = pgnet.model(X)
        # print("new Action Y: ", np.round(new_action_Y, 4))
        # print("new Value Y: ", new_value_Y)

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

    print("A Mean: ", np.round(np.mean(output_A, axis=0), 4))
    print("V Mean: ", np.round(np.mean(output_V, axis=0), 4))

    print("A Max: ", np.round(np.max(np.abs(output_A), axis=0), 4))
    print("V MAx: ", np.round(np.max(np.abs(output_V), axis=0), 4))


if __name__ == "__main__":
    test_action_value_fit()