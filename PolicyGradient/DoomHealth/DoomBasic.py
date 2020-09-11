# Just disables the warning, doesn't enable AVX/FMA
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES']= '-1'


import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
from vizdoom import *        # Doom Environment

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

import random                # Handling random number generation
import time                  # Handling time calculation
from skimage import transform# Help us to preprocess the frames
import datetime
import time
from ConvPGNet import ConvPGNet
from disp_multiple_images import show_images

from collections import deque# Ordered collection with ends

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')

print(tf.__version__)

action_names = ['turn_left', 'turn_right', 'move_forward']

def create_environment():
    game = DoomGame()

    # Load the config file
    game.load_config("basic.cfg")

    # Load the correct scenario
    game.set_doom_scenario_path("basic.wad")

    game.init()

    # define actions
    possible_actions = [[1 if i == j else 0 for i in range(game.get_available_buttons_size())] for j in range(game.get_available_buttons_size())]

    return game, possible_actions

def test_environment():
    game, possible_actions = create_environment()

    episodes = 10
    for i in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            misc = state.game_variables
            action = random.choice(possible_actions)
            action_name = action_names[np.argmax(action)]
            print(action_name)
            reward = game.make_action(action)
            print("\tReward:", reward)
            time.sleep(0.02)
        print ("Result:", game.get_total_reward())
        time.sleep(2)
    game.close()

# crop image to remove ceiling and change size 
def preprocess_frame(frame):
    # Greyscale frame already done in our vizdoom config
    # x = np.mean(frame,-1)

    # Crop the screen (remove the roof because it contains no information)
    cropped_frame = frame[30:-10, 30:-30]

    # Normalize Pixel Values
    normalized_frame = cropped_frame/255.0

    # Resize
    preprocessed_frame = transform.resize(normalized_frame, [84, 84])
    return preprocessed_frame

# create state from a stack of four frames to picture movement
# Parameters: Deque of frames, new frame to be added
# Output: Updated deque of frames with new frame, np array stack of updated deque

stack_length = 4

def stack_frames(frames_deque, frame, is_new_episode):

    if is_new_episode:
        frames_deque = [frame for i in range(stack_length)]
        frames_deque = deque(frames_deque, maxlen=stack_length)
    else:
        frames_deque.append(frame)

    # The neural network will take in (B, H, W, in_channels)
    # Stack on Axis 2 will be the in_channels
    stacked_state = np.stack(frames_deque, axis=2)

    return frames_deque, stacked_state

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


# HYPERPARAMETERS ===========================================================================
state_size = [84, 84, 4]
num_actions = 3

# Training parameters
max_epochs = 500
epoch_save = 5
gamma = 0.95
batch_size = 1000

# Model Parameters
learning_rate = 1e-4

# ===========================================================================================


def get_batch(action_policy, batch_size=500, verbose=False):
    game, possible_actions = create_environment()
    total_rewards = []
    num_actions = game.get_available_buttons_size()
    current_deque = deque()

    total_reward_count = 0
    episode_count = 0

    states_mb, actions_mb, rewards_mb, discounted_rewards_mb = [], [], [], []

    while total_reward_count < batch_size:
        game.new_episode()
        episode_states = []
        episode_actions = []
        episode_rewards = []

        done = False

        while not done:
            img = game.get_state().screen_buffer
            current_deque, current_state = stack_frames(current_deque, preprocess_frame(img), len(episode_states)==0)

            action_prob = np.array(action_policy.model(tf.reshape(current_state, [-1, *state_size])))
            # print(action_prob)
            action_index = np.random.choice(range(num_actions), p=action_prob[0])
            action = possible_actions[action_index]

            reward = game.make_action(action)
            total_reward_count += 1

            episode_states.append(current_state)
            episode_actions.append([1 if i == action_index else 0 for i in range(num_actions)])
            episode_rewards.append(reward)

            done = game.is_episode_finished()

        # Lists need to be np.arrays to be processed by model

        episode_states = np.array(episode_states)
        episode_actions = np.array(episode_actions)
        discounted_rewards = discount_and_normalize_rewards(episode_rewards)
        episode_rewards = np.array(episode_rewards)

        states_mb.append(episode_states)
        actions_mb.append(episode_actions)
        rewards_mb.append(episode_rewards)
        discounted_rewards_mb.append(discounted_rewards)

        total_ep_reward = np.sum(episode_rewards)
        total_rewards.append(total_ep_reward)

        if verbose:
            print("Episode {} Reward: {}".format(episode_count+1, total_ep_reward))
        episode_count += 1

    states_mb = np.concatenate(states_mb)
    actions_mb = np.concatenate(actions_mb)
    rewards_mb = np.concatenate(rewards_mb)
    discounted_rewards_mb = np.concatenate(discounted_rewards_mb)

    if verbose:
        print("States_mb: {}".format(states_mb.shape))
        print("Actions_mb: {}".format(actions_mb.shape))
        print("rewards_mb: {}".format(rewards_mb.shape))
        print("Discounted_rewards_mb: {}".format(discounted_rewards_mb.shape))
        print("Episode Rewards: {}".format(total_rewards))
        print("Max Reward: {}".format(np.max(total_rewards)))
        print("Average Reward: {}".format(np.mean(total_rewards)))

    game.close()

    return states_mb, actions_mb, rewards_mb, discounted_rewards_mb, total_rewards

if __name__ == "__main__":

    start = time.time()

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    action_policy = ConvPGNet(state_size, num_actions=num_actions, learning_rate=learning_rate)

    for i_epoch in range(max_epochs):
        states_mb, actions_mb, rewards_mb, discounted_rewards_mb, total_rewards = get_batch(action_policy, batch_size, verbose=False)

        loss = action_policy.fit_gradient(states_mb, actions_mb, discounted_rewards_mb, train_summary_writer, i_epoch)
        print("===============================================")
        print("Epoch: {} / {}".format(i_epoch+1, max_epochs))
        print("------------")
        print("Number of training episodes: {}".format(len(total_rewards)))
        print("Total Reward: {}".format(np.sum(total_rewards)))
        print("Average Reward of batch: {}".format(np.mean(total_rewards)))
        print("Training loss: {}".format(loss))
        
        with train_summary_writer.as_default():
            tf.summary.scalar('Loss', loss, step=i_epoch)
            tf.summary.scalar('TotalReward', np.sum(total_rewards), step=i_epoch)
            tf.summary.scalar('AverageReward', np.mean(total_rewards), step=i_epoch)
            for i, layer in enumerate(action_policy.model.layers):
                if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
                    w = layer.get_weights()[0]
                    b = layer.get_weights()[1]
                    tf.summary.histogram("Layer"+str(i)+"Weights", w, step=i_epoch)
                    tf.summary.histogram("Layer"+str(i)+"Bias", b, step=i_epoch)
                    tf.summary.scalar("Layer"+str(i)+"WeightsMean", np.mean(w), step=i_epoch)
                    tf.summary.scalar("Layer"+str(i)+"BiasMean", np.mean(b), step=i_epoch)
                    tf.summary.scalar("Layer"+str(i)+"WeightsStd", np.std(w), step=i_epoch)
                    tf.summary.scalar("Layer"+str(i)+"BiasStd", np.std(b), step=i_epoch)
        
        if i_epoch % epoch_save == 0:
            action_policy.model.save_weights('checkpoint/my_checkpoint')

    print("Total Runtime: %s seconds" % (time.time()-start))