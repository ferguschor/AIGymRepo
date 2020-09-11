import numpy as np
import SonicEnv
from SimplePGNet import SimplePGNet
from baselines.common.vec_env import SubprocVecEnv
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense
from ModelStatsBoard import WeightWriter, TestRewardWriter
import gym
import datetime
from tensorboard.plugins.hparams import api as hp
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES']= '-1'

class CartEnvRunner:

    def __init__(self, env, model, nsteps, gamma):
        self.env = env
        if hasattr(env, 'num_envs'):
            self.nenv = env.num_envs
        else:
            self.nenv = 1
        self.model = model
        self.nsteps = nsteps
        self.batch_obs_shape = (self.nenv * self.nsteps,) + env.observation_space.shape
        # Obs stores the latest observation
        self.obs = np.zeros((self.nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype)
        self.obs[:] = env.reset()
        self.dones = np.array([False for _ in range(self.nenv)])
        self.gamma = gamma

    def run(self):

        num_actions = self.env.action_space.n

        states_mb = []
        # Actions should be one-hot encoded vectors
        actions_mb = []
        values_mb = []
        rewards_mb = []
        next_state_mb = []
        dones_mb = []

        for i_step in range(self.nsteps):

            sel_action_list = []
            value_list = []
            # state = np.reshape(state, (-1, *state.shape))  # (N, 4)
            action_probs, value = self.model.model(self.obs)  # (N, 2), (N, 1)
            action_probs = np.array(action_probs)
            # Renormalize in case there is a small error
            action_probs = action_probs / np.sum(action_probs, axis=1)[:, None]  # (N, 2), sum removes an axis, need to add it back

            sel_action_list = [np.random.choice(range(num_actions), p=prob) for prob in action_probs]
            # sel_action = np.random.choice(range(num_actions), p=action_probs)
            # sel_action_list.append(sel_action)

            # Value ouptut is of the shape (N, 1) but the list requires only the scalar
            value_list = np.squeeze(np.array(value), axis=1)  # (N, )

            # Self.obs contains the current states
            states_mb.append(self.obs)  # s_t

            # One-hot vector encode actions
            action_vector = np.zeros((self.nenv, num_actions), dtype=np.uint8)
            action_vector[range(self.nenv), sel_action_list] = 1
            actions_mb.append(action_vector)  # a_t

            values_mb.append(value_list)   # v_t
            dones_mb.append(self.dones)  # done_t

            # Reward and done to be appended after the step
            next_states, rews, dones, infos = self.env.step(sel_action_list)
            self.env.render()

            rewards_mb.append(rews)  # r_t
            next_state_mb.append(next_states)  # s_t+1
            self.obs = next_states
            self.dones = dones

        # Need to append one more time
        dones_mb.append(dones)  # done_t

        # Keep mb's as list for processing

        #   To process we need to calculate deltas and advantages
        #   Advantages are the sum of discounted deltas up to the end of the episode / nsteps
        #   Delta = R(t) + gamma * V(s_t+n) - V(s_t)

        # Discount / bootstrap off value fns
        # Self.obs and self.done hold the final observations and dones
        last_value = self.model.model(self.obs)[1]
        last_value = np.squeeze(last_value, axis=1)
        next_dones_mb = dones_mb[1:]

        # ... D_t+n-2   D_t+n-1     D_t+n   D_t+n+1     if D_t+n+1 == 0, need to use V
        # ... R_t+n-2   R_t+n-1     R_t+n               if D_t+n+1 == 1, then 0
        assert last_value.shape == self.dones.shape, (last_value.shape, self.dones.shape)
        last_value = (1.-self.dones) * last_value.reshape(self.dones.shape)

        #   Adv = R(t) + gamma * R(t+1) + gamma ^ 2 * R(t+2) + ... + gamma ^ n * V(t+n)
        #   Last element of array would have r = V, which we do not need
        discounted_rewards = discount_rewards(rewards_mb + [last_value], dones_mb + [self.dones], self.gamma)[:-1]
        # Convert mb's to np.array
        states_mb = np.array(states_mb)
        actions_mb = np.array(actions_mb)
        values_mb = np.array(values_mb)
        rewards_mb = np.array(discounted_rewards)
        next_state_mb = np.array(next_state_mb)
        dones_mb = np.array(dones_mb)
        next_dones_mb = np.array(next_dones_mb)

        # return states_mb, actions_mb, rewards_mb, next_state_mb, dones_mb
        return map(swapflatten01, (states_mb, actions_mb, values_mb, rewards_mb, next_dones_mb))


# Given a list of trajectory rewards, return the discounted rewards for each time step as list
def discount_rewards(rewards, dones, gamma):
    discounted_rewards = []
    r = np.zeros(rewards[0].shape)

    # Start from end of list and discount
    for rew, done in zip(rewards[::-1], dones[::-1]):
        r = rew + gamma * r * (1.-done)
        # First element in list will correspond to the last element in parameter list
        discounted_rewards.append(r)

    # Return reversed to correspond with original list
    return discounted_rewards[::-1]


def swapflatten01(arr):
    # Arr is shape (Steps, Nenv, H, W, Frames)
    # Swap axes to group by rollout / trajectory instead of by step
    arr = np.swapaxes(arr, 0, 1)
    s = arr.shape
    # Flatten the first two axes to create a coherent batch
    arr = np.reshape(arr, (s[0] * s[1], *s[2:]))
    return arr


def main(hParams, n_run, total_timesteps):
    nsteps = hParams['N_STEPS']
    nenv = hParams[HP_N_ENV]
    n_epochs = hParams['N_EPOCHS']
    # total_timesteps = int(n_epochs * nsteps * nenv)
    nbatch = nenv * nsteps
    n_train = 4
    n_minibatch = 8
    minibatch_size = nbatch // n_minibatch

    update_int = 1
    save_int = 5
    test_int = 10

    gamma = 0.99 * 0.95
    lr = hParams[HP_LEARNING_RATE]
    vf_coef = hParams[HP_VF_COEF]
    ent_coef = hParams[HP_ENT_COEF]
    clip_param = 0.2
    save_dir = 'lr' + str(lr) + 'vc' + str(vf_coef) + 'ec' + str(ent_coef) + 'env' + str(nenv)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/test_hparam_tuning/run-' + str(n_run)
    summ_writer = tf.summary.create_file_writer(log_dir)

    envfn = lambda: gym.make('CartPole-v0')
    env = SubprocVecEnv([envfn] * nenv)
    state_size = env.observation_space.shape
    num_actions = env.action_space.n
    current_net = SimplePGNet(state_size, num_actions, learning_rate=lr, vf_coef=vf_coef, ent_coef=ent_coef, clip_param=0.2)
    old_net = SimplePGNet(state_size, num_actions, learning_rate=lr, vf_coef=vf_coef, ent_coef=ent_coef, clip_param=0.2)

    runner = CartEnvRunner(env, current_net, nsteps, gamma)

    print("Total updates to run: ", total_timesteps // nbatch)
    for update in range(1, total_timesteps // nbatch + 1):

        print("\nUpdate #{}:".format(update))
        states_mb, actions_mb, values_mb, rewards_mb, next_dones_mb = runner.run()

        for _ in range(n_train):
            indices = np.arange(nbatch)

            np.random.shuffle(indices)

            for start in range(0, nbatch, minibatch_size):
                end = start + minibatch_size
                bind = indices[start:end]

                old_action_prob_mb, old_values_mb = old_net.model(states_mb[bind])
                # Copy current model to old model since current model needs to be updated
                # Old model will always be one step behind current model
                old_net.model.set_weights(current_net.model.get_weights())

                policy_loss, entropy_loss, vf_loss, loss = current_net.fit_gradient(states_mb[bind], actions_mb[bind], rewards_mb[bind], values_mb[bind], old_action_prob_mb, old_values_mb)


        WeightWriter(summ_writer, current_net, (Conv2D, Dense), global_step=update)

        with summ_writer.as_default():
            tf.summary.scalar("PolicyLoss", policy_loss, step=update)
            tf.summary.scalar("EntropyLoss", entropy_loss, step=update)
            tf.summary.scalar("ValueFunctionLoss", vf_loss, step=update)
            tf.summary.scalar("Loss", loss, step=update)

        if update % update_int == 0:
            print("PolicyLoss:", policy_loss)
            print("EntropyLoss: ", entropy_loss)
            print("ValueFunctionLoss: ", vf_loss)
            print("Loss: ", loss)

        if update % save_int == 0:
            current_net.model.save_weights('test_hparams_tuning_models/' + save_dir + '/my_checkpoint')
            print("Model Saved")

        if update % test_int == 0:
            TestRewardWriter(summ_writer, envfn, current_net, 20, global_step=update)

    with summ_writer.as_default():
        hp.hparams(hParams)

    env.close()


def test_agent(n_episodes=100, load_dir='checkpoint/my_checkpoint', verbose=False):
    nenv = 1

    envfn = lambda: gym.make('CartPole-v0')
    env = envfn()

    state_size = env.observation_space.shape
    num_actions = env.action_space.n
    lr = 2e-4
    vf_coef = 0.0
    ent_coef = 0.0

    pgnet = SimplePGNet(state_size, num_actions, learning_rate=lr, vf_coef=vf_coef, ent_coef=ent_coef)
    pgnet.model.load_weights(load_dir)

    obs = env.reset()

    total_reward_list = []

    for i_episode in range(n_episodes):
        done = False
        total_reward = 0
        while not done:
            obs = np.reshape(obs, (-1, *state_size))
            action_probs = np.array(pgnet.model(obs)[0])
            sel_action = np.argmax(action_probs)
            obs, rew, done, info = env.step(sel_action)
            total_reward += rew
        if verbose: print("Episode {} Reward: {}".format(i_episode+1, total_reward))
        total_reward_list.append(total_reward)
        obs = env.reset()
    total_reward_list = np.array(total_reward_list)

    env.close()

    sum_reward = np.sum(total_reward_list)
    avg_reward = np.mean(total_reward_list)
    max_reward = np.max(total_reward_list)
    if verbose: print("Total Reward: {}".format(sum_reward))
    if verbose: print("Average Reward: {}".format(avg_reward))
    if verbose: print("Max Reward: {}".format(max_reward))

    return sum_reward, avg_reward, max_reward


if __name__ == "__main__":
    HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.005]))
    HP_VF_COEF = hp.HParam('vf_coefficient', hp.Discrete([1.0]))
    HP_ENT_COEF = hp.HParam('entropy_coefficient', hp.Discrete([0.0]))
    HP_N_ENV = hp.HParam('n_envrionments', hp.Discrete([1]))
    n_run = 0

    for _ in range(50):
        for learning_rate in HP_LEARNING_RATE.domain.values:
            for vf_coef in HP_VF_COEF.domain.values:
                for ent_coef in HP_ENT_COEF.domain.values:
                    for n_env in HP_N_ENV.domain.values:
                        hparams = {
                            HP_LEARNING_RATE: learning_rate,
                            HP_VF_COEF: vf_coef,
                            HP_ENT_COEF: ent_coef,
                            HP_N_ENV: n_env,
                            'N_STEPS': 250,
                            'N_EPOCHS': 300,
                            'RUN_NUMBER': n_run
                        }
                        main(hparams, n_run, total_timesteps=250*300)
                        n_run += 1

    # test_agent(load_dir='lr0.0001_models/lr0.0001vc0.25ec0.05env5/my_checkpoint', verbose=True)



