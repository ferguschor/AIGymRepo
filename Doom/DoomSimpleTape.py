# Just disables the warning, doesn't enable AVX/FMA
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES']= '-1'

import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
from vizdoom import *        # Doom Environment

import random                # Handling random number generation
import time                  # Handling time calculation
from skimage import transform# Help us to preprocess the frames
import datetime
import time

from DQNetwork import DQNetwork, Memory
from disp_multiple_images import show_images

from collections import deque# Ordered collection with ends
import matplotlib.pyplot as plt # Display graphs

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')

from PIL import Image

print(tf.__version__)

def create_environment():
    game = DoomGame()

    # Load the config file
    game.load_config("C:/Users/Fergus/PycharmProjects/AIGym/AIGymRepo/Doom/basic.cfg")

    # Load the correct scenario
    game.set_doom_scenario_path("C:/Users/Fergus/PycharmProjects/AIGym/AIGymRepo/Doom/basic.wad")

    game.init()

    # define actions
    left = [1,0 ,0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]

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
            print(action)
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
    cropped_frame = frame[30:-10,30:-30]

    # Normalize Pixel Values
    normalized_frame = cropped_frame/255.0

    # Resize
    preprocessed_frame = transform.resize(normalized_frame, [84,84])

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


# Run the game to fill memory with random SARSA tuples
def pretrain(pretrain_length):
    game, possible_actions = create_environment()


    current_state_deque = deque()
    action_names = ['Left', 'Right', 'Shoot']

    # Add to memory with random actions

    game.new_episode()

    # Grab initial state for new episode
    state = game.get_state()
    img = state.screen_buffer
    current_state_deque, current_state = stack_frames(current_state_deque, preprocess_frame(img), True)

    # Fill memory with states from random actions for pretraining
    for i in range(pretrain_length):

        # Select random action
        action = random.choice(possible_actions)
        print(action_names[np.argmax(action)])
        reward = game.make_action(action)
        print ("\tReward:", reward)

        if game.is_episode_finished():
            # Can't grab state when the episode is done
            next_state = np.zeros(state_size)
            # Save SARSA experience
            experience.append((current_state, action, reward, next_state, game.is_episode_finished()))

            game.new_episode()
            # Grab initial state for new episode
            state = game.get_state()
            img = state.screen_buffer
            current_state_deque, current_state = stack_frames(current_state_deque, preprocess_frame(img), True)

        else:
            # Grab state' after action
            state = game.get_state()
            img = state.screen_buffer
            current_state_deque, next_state = stack_frames(current_state_deque, preprocess_frame(img), False)
            # Save SARSA experience
            experience.append((current_state, action, reward, next_state, game.is_episode_finished()))
            # Update the old state
            current_state = next_state

        # time.sleep(0.02)

        print("Result:", game.get_total_reward())
        print("Deque length: ",len(current_state_deque))
        # Shows last four frames
        # show_images(current_state_deque)
        # time.sleep(2)
    game.close()

# HYPERPARAMETERS----------------------------------------------------------------------------------------------------
state_size = [84,84, 4]
memory_size = 100000
experience = Memory(memory_size=memory_size)
batch_size = 64
total_episodes = 100
learning_rate = 0.0001
discount_rate = 0.95

action_names = ['Left', 'Right', 'Shoot']

checkpoint_path = "C:/Users/Fergus/PycharmProjects/AIGym/AIGymRepo/Doom/checkpoint/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# -------------------------------------------------------------------------------------------------------------------

# temporal difference training
def td_train(load_weights=False):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'C:/Users/Fergus/PycharmProjects/AIGym/AIGymRepo/Doom/logs/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)


    game, possible_actions = create_environment()
    current_state_deque = deque()

    max_steps = 100
    epsilon = 1.0 # Chance of taking a random action
    min_epsilon = 0.01
    max_epsilon = 1.0
    decay_rate = 0.0001 # epsilon decay rate per step
    decay_step = 0


    DQN = DQNetwork(state_size=state_size, num_actions=game.get_available_buttons_size(), learning_rate=learning_rate)

    if load_weights:
        DQN.model.load_weights('C:/Users/Fergus/PycharmProjects/AIGym/AIGymRepo/Doom/loadcheckpoint/my_checkpoint')

    # take action (epsilon random)
    # for each action, save SARSA into memory
    # take sample batch
    # create target y's
    # train model on batch
    # repeat for each set steps

    episode_rewards = []


    for episode in range(total_episodes):

        step = 0
        game.new_episode()

        img = game.get_state().screen_buffer
        current_state_deque, current_state = stack_frames(current_state_deque, preprocess_frame(img), True)

        # Fo every step, a new memory is added and the model is trained on a batch
        while step < max_steps:
            step += 1

            if np.random.uniform(0,1) < epsilon:
                action = random.choice(possible_actions)
            else:
                action_index = np.argmax(DQN.model.predict(np.reshape(current_state, [-1, *state_size])), axis=1)[0]
                # print('Network Action')
                action = possible_actions[action_index]

            decay_step += 1
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * decay_step)


            print(action_names[np.argmax(action)])
            print(action)
            reward = game.make_action(action)
            # print ("\tNew Reward:", reward)

            if game.is_episode_finished():
                # Can't grab state when the episode is done
                next_state = np.zeros(state_size)
                # Save SARSA experience
                experience.append((current_state, action, reward, next_state, game.is_episode_finished()))
                step = max_steps

            else:
                # Grab state' after action
                state = game.get_state()
                img = state.screen_buffer
                current_state_deque, next_state = stack_frames(current_state_deque, preprocess_frame(img), False)
                # Save SARSA experience
                experience.append((current_state, action, reward, next_state, game.is_episode_finished()))
                # Update the old state
                current_state = next_state

            training_batch = experience.sample(batch_size)
            current_state_batch = np.array([each[0] for each in training_batch])
            action_batch = np.array([each[1] for each in training_batch])
            reward_batch = np.array([each[2] for each in training_batch])
            next_state_batch = np.array([each[3] for each in training_batch])
            done_batch = np.array([each[4] for each in training_batch])

            current_Q = DQN.model.predict(current_state_batch)
            next_Q = np.max(DQN.model.predict(next_state_batch), axis=1)

            # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
            target_Q_batch = []
            for i in range(len(training_batch)):
                if done_batch[i]:
                    target_Q_batch.append(reward_batch[i])
                else:
                    target_Q_batch.append(reward_batch[i] + discount_rate * np.max(next_Q[i]))

            # Training on a single batch
            DQN.single_action_train(current_state_batch, action_batch, target_Q_batch)

        with train_summary_writer.as_default():
            tf.summary.scalar('Loss', DQN.train_loss.result(), step=episode)

        print("Episode {} Reward: {} Loss: {}".format(episode+1,game.get_total_reward(), DQN.train_loss.result()))
        episode_rewards.append(game.get_total_reward())

        DQN.train_loss.reset_states()

        DQN.model.save_weights('C:/Users/Fergus/PycharmProjects/AIGym/AIGymRepo/Doom/checkpoint/my_checkpoint')
        print("Model Saved")

    print("Average Episode Reward:", np.mean(episode_rewards))

    game.close()


def test_agent():
    game, possible_actions = create_environment()

    DQN = DQNetwork(state_size=state_size, num_actions=game.get_available_buttons_size(), learning_rate=learning_rate)
    DQN.model.load_weights('C:/Users/Fergus/PycharmProjects/AIGym/AIGymRepo/Doom/loadcheckpoint/my_checkpoint')

    current_state_deque = deque()

    episode_rewards = []

    for i in range(10):

        game.new_episode()
        img = game.get_state().screen_buffer
        current_state_deque, current_state = stack_frames(current_state_deque, preprocess_frame(img), True)

        while not game.is_episode_finished():

            action_index = np.argmax(DQN.model.predict(np.reshape(current_state, [-1, *state_size])), axis=1)[0]
            action = possible_actions[action_index]

            reward = game.make_action(action)
            print("Action: ", action_names[action_index])

            if game.is_episode_finished():
                break
            else:
                img = game.get_state().screen_buffer
                current_state_deque, next_state = stack_frames(current_state_deque, preprocess_frame(img), False)
                current_state = next_state
            time.sleep(0.02)

        episode_rewards.append(game.get_total_reward())
        print("Reward: ", game.get_total_reward())

    print("Average Episode Reward:", np.mean(episode_rewards))
    game.close()


def output_check_single(sample_size=5):
    pretrain(pretrain_length=batch_size)

    DQN = DQNetwork(state_size=state_size, num_actions=3, learning_rate=learning_rate)
    DQN.model.load_weights('C:/Users/Fergus/PycharmProjects/AIGym/AIGymRepo/Doom/loadcheckpoint/my_checkpoint')

    for i in experience.sample(sample_size):
        state = i[0]
        action = i[1]
        reward = i[2]
        next_state = i[3]
        done = i[4]
        print("\nMeans: ", np.mean(state, axis=(0, 1)))
        print("St.devs: ", np.std(state, axis=(0, 1)))
        output = DQN.model(np.reshape(state, [-1, *state_size]))
        next_output = DQN.model(np.reshape(next_state, [-1, *state_size]))
        if done:
            target = reward
        else:
            target = reward + discount_rate * np.max(next_output, axis=1)

        print("Model Output: ", output)
        print("Model Next Output: ", next_output)
        print("Target: ", target)
        print("Action: ", action)
        print("Multiply: ", tf.multiply(output, action))
        current_Q = tf.reduce_sum(tf.multiply(output, action), axis=1)
        print("Current Q: ", current_Q)
        print("Loss: ", tf.keras.losses.mean_squared_error(current_Q, target))
        print(state.shape)

def output_check_batch(sample_size=5):
    pretrain(pretrain_length=200)

    DQN = DQNetwork(state_size=state_size, num_actions=3, learning_rate=learning_rate)
    DQN.model.load_weights('C:/Users/Fergus/PycharmProjects/AIGym/AIGymRepo/Doom/loadcheckpoint/my_checkpoint')

    loss = 0

    while loss < 100:
        training_batch = experience.sample(sample_size)
        current_state_batch = np.array([each[0] for each in training_batch])
        action_batch = np.array([each[1] for each in training_batch])
        reward_batch = np.array([each[2] for each in training_batch])
        next_state_batch = np.array([each[3] for each in training_batch])
        done_batch = np.array([each[4] for each in training_batch])

        current_output = DQN.model.predict(current_state_batch)
        print("Current Q: ", current_output)
        print("Action: ", action_batch)
        print("Action Mult: ", tf.multiply(current_output, action_batch))
        print("Reduce Sum: ", tf.reduce_sum(tf.multiply(current_output, action_batch), axis=1))
        current_Q = tf.reduce_sum(tf.multiply(current_output, action_batch), axis=1)
        next_Q = DQN.model.predict(next_state_batch)
        print("Next Q: ", next_Q)
        next_Q_action = np.max(DQN.model.predict(next_state_batch), axis=1)
        print("Next Q Max: ", next_Q_action)

        target_Q_batch = []
        for i in range(len(training_batch)):
            if done_batch[i]:
                target_Q_batch.append(reward_batch[i])
            else:
                target_Q_batch.append(reward_batch[i] + discount_rate * np.max(next_Q[i]))

        print("Target Q: ", target_Q_batch)
        print("Test Loss: ", tf.reduce_mean(tf.square(target_Q_batch-current_Q)))
        loss = tf.keras.losses.mean_squared_error(y_true=target_Q_batch, y_pred=current_Q)
        print("Loss: ", loss)

if __name__ == "__main__":

    start = time.time()

    # pretrain(pretrain_length=batch_size)
    # td_train(load_weights=False)


    # DQN = DQNetwork(state_size=state_size, num_actions=3, learning_rate=learning_rate)
    # DQN.model.load_weights('C:/Users/Fergus/PycharmProjects/AIGym/AIGymRepo/Doom/loadcheckpoint/my_checkpoint')

    output_check_batch(batch_size)

    # test_agent()

    # print("Total Memory States: ", len(experience))


    # output = DQN.model.predict(img)
    # print(img)
    # print("OUTPUT: ", output)
    #
    # print(DQN.model.trainable_variables)
    # print("Total Runtime: %s seconds" % (time.time()-start))