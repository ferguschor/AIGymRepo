import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import datetime
import time
from SimplePGNet import SimplePGNet

# HYPERPARAMETERS --------------------------------------------------------------

# Model Parameters
state_size = [4]
learning_rate = 0.01


# Training Parameters
total_episodes = 400
episode_save = 5

# Value Parameters
gamma = 0.95


def discount_and_normalize_rewards(episode_rewards):
    # Calculate cumulative discounted rewards from the current t = 0
    # Discounted_rewards[i] = expected reward at step i
    discounted_rewards = np.zeros((len(episode_rewards)))
    cumulative_reward = 0.0

    for i in reversed(range(len(episode_rewards))):
        cumulative_reward = cumulative_reward * gamma + episode_rewards[i]
        discounted_rewards[i] = cumulative_reward

    mean = np.mean(discounted_rewards)
    std = np.std(discounted_rewards)
    discounted_rewards = (discounted_rewards - mean) / std
    return discounted_rewards


def train():
    env = gym.make('CartPole-v0')
    env.seed(1)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'C:/Users/Fergus/PycharmProjects/AIGym/AIGymRepo/PolicyGradient/logs/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    state_size = env.observation_space.shape
    num_actions = env.action_space.n

    SPG = SimplePGNet(state_size, num_actions, learning_rate)

    total_rewards = []
    total_losses = []

    actions_index = range(env.action_space.n)

    for i_episode in range(total_episodes):
        observation = env.reset()

        step = 0
        done = False
        chosen_actions_history = []
        rewards_history = []
        states_history = [observation]

        while not done:
            step += 1

            env.render()
            action_prob = SPG.model(tf.reshape(observation, [-1, *state_size]))
            action_prob = np.array(action_prob)
            # action_prob is a list of lists, take first element for action probs
            action = np.random.choice(actions_index, p=action_prob[0])

            observation, reward, done, info = env.step(action)

            # Record for training
            rewards_history.append(reward)
            action_as_list = [1 if i == action else 0 for i in actions_index]
            chosen_actions_history.append(action_as_list)
            states_history.append(observation)

            if done:
                total_episode_reward = np.sum(rewards_history)
                total_rewards.append(total_episode_reward)
                print("========================================================")
                print("Episode {} finished after {} timesteps".format(i_episode+1, step))
                print("Reward: {}".format(total_episode_reward))
                # break
    
        discounted_rewards = discount_and_normalize_rewards(rewards_history)
        states_history = np.array(states_history[:-1]) # last state has no following action
        chosen_actions_history = np.array(chosen_actions_history)
        discounted_rewards = np.array(discounted_rewards)

        loss = SPG.fit_gradient(states_history, chosen_actions_history, discounted_rewards)
        total_losses.append(loss)
        print("Max Reward so far: {}".format(np.max(total_rewards)))
        
        with train_summary_writer.as_default():
            tf.summary.scalar('Loss', loss, step=i_episode)
            tf.summary.scalar('Reward', total_episode_reward, step=i_episode)
            tf.summary.scalar('MaxRewardObserved', np.max(total_rewards), step=i_episode)
            tf.summary.scalar('AvgReward', np.mean(total_rewards), step=i_episode)
            for i, layer in enumerate(SPG.model.layers):
                if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
                    w = layer.get_weights()[0]
                    b = layer.get_weights()[1]
                    tf.summary.histogram("Layer"+str(i)+"Weights", w, step=i_episode)
                    tf.summary.histogram("Layer"+str(i)+"Bias", b, step=i_episode)
                    tf.summary.scalar("Layer"+str(i)+"WeightsMean", np.mean(w), step=i_episode)
                    tf.summary.scalar("Layer"+str(i)+"BiasMean", np.mean(b), step=i_episode)
                    tf.summary.scalar("Layer"+str(i)+"WeightsStd", np.std(w), step=i_episode)
                    tf.summary.scalar("Layer"+str(i)+"BiasStd", np.std(b), step=i_episode)
        
        if i_episode % episode_save == 0:
            SPG.model.save_weights('C:/Users/Fergus/PycharmProjects/AIGym/AIGymRepo/PolicyGradient/checkpoint/my_checkpoint')

    print("Max Episode Reward: {}".format(np.max(total_rewards)))
    print("Average Episode Reward: {}".format(np.mean(total_rewards)))
    print("Std Episode Reward: {}".format(np.std(total_rewards)))
    env.close()

def test_agent():
    env = gym.make('CartPole-v0')
    env.seed(42)

    total_rewards = []

    SPG = SimplePGNet(state_size, num_actions=env.action_space.n, learning_rate=learning_rate)
    SPG.model.load_weights('C:/Users/Fergus/PycharmProjects/AIGym/AIGymRepo/PolicyGradient/checkpoint/my_checkpoint')


    for i_episode in range(100):
        episode_reward = 0

        observation = env.reset()
        done = False

        while not done:

            env.render()
            input = tf.reshape(observation, [-1, *state_size])
            action = np.argmax(np.array(SPG.model(input))[0])

            observation, reward, done, info = env.step(action)

            episode_reward += reward

            if done:
                print("Episode Reward: {}".format(episode_reward))
                total_rewards.append(episode_reward)

    print("Average Episode Rewards: {}".format(np.mean(total_rewards)))


if __name__=="__main__":

    start = time.time()

    train()
    # test_agent()

    print("Total Runtime: {}".format(time.time()-start))

    # pg = SimplePGNet(state_size, learning_rate)
    #
    # inputs = [0.2, 0.5, 0.12, 0.89]
    # inputs = tf.reshape(inputs, [-1, *state_size])
    # print("Inputs: {}".format(inputs))
    # outputs = pg.model(inputs)
    # print('Outputs: {}'.format(outputs))
    #
    # actions_gen = np.random.randint(0, 2, 10)
    # actions = np.array([[i, 1-i] for i in actions_gen])
    # print(actions)
    #
    # neg_log_prob = tf.keras.losses.categorical_crossentropy(y_true=actions, y_pred=outputs)
    #         # loss = tf.reduce_mean(tf.multiply(neg_log_prob, discounted_episode_rewards))
    #
    # print("Log Prob: {}".format(neg_log_prob))
    #
    # formula = tf.reduce_sum(actions * -np.log(outputs), axis=1)
    # print("Formulas: {}".format(formula))
    #
    # env = gym.make('CartPole-v0')
    # env.seed(1)
    #
    # action = np.random.choice([0, 1])
    # print(action)

