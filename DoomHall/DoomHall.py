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
from PERMemory import Memory

from DuelDQNetwork import DuelDQNetwork
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
    game.load_config("deadly_corridor.cfg")

    # Load the correct scenario
    game.set_doom_scenario_path("deadly_corridor.wad")

    game.init()

    # define actions
    action_names = ['move_left', 'move_right', 'shoot', 'move_forward', 'move_backward', 'turn_left', 'turn_right']
    possible_actions = [[1 if i == j else 0 for i in range(game.get_available_buttons_size())] for j in range(game.get_available_buttons_size())]

    return game, possible_actions

def test_environment():
    game, possible_actions = create_environment()

    episodes = 1
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
    cropped_frame = frame[15:-5,20:-20]

    # Normalize Pixel Values
    normalized_frame = cropped_frame/255.0

    # Resize
    preprocessed_frame = transform.resize(normalized_frame, [100,120])
    preprocessed_frame = preprocessed_frame.astype(np.float32)
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
    # Add to memory with random actions

    game.new_episode()

    # Grab initial state for new episode
    state = game.get_state()
    img = state.screen_buffer
    current_state_deque, current_state = stack_frames(current_state_deque, preprocess_frame(img), True)

    # Fill memory with states from random actions for pretraining
    for i in range(pretrain_length):

        print("\nPretrain Step: {}".format(i+1))

        # Select random action
        action = random.choice(possible_actions)
        print(action_names[np.argmax(action)])
        reward = game.make_action(action)
        print ("\tReward:", reward)

        done = game.is_episode_finished()
        if done:
            # Can't grab state when the episode is done
            blank_img = np.zeros((state_size[0], state_size[1]), dtype=np.int)
            current_state_deque, next_state = stack_frames(current_state_deque, preprocess_frame(blank_img), False)

            # Save SARSA experience
            experience.store((current_state, action, reward, next_state, done))

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
            experience.store((current_state, action, reward, next_state, done))
            # Update the old state
            current_state = next_state

        # time.sleep(0.02)

        print("Result:", game.get_total_reward())
        # Shows last four frames
        # show_images(current_state_deque)
        # time.sleep(2)
    game.close()

# HYPERPARAMETERS----------------------------------------------------------------------------------------------------

# Model Parameters
state_size = [100, 120, 4]
action_names = ['move_left', 'move_right', 'shoot', 'move_forward', 'move_backward', 'turn_left', 'turn_right']
num_actions = len(action_names)
learning_rate = 0.00025

# Memory Buffer
memory_size = 100
experience = Memory(capacity=memory_size)

# Training Parameters
batch_size = 64
total_episodes = 5000
max_steps = 5000 # per episode
target_update = 1
pre_train_length = 100
episode_save = 5

# Learning Parameters
discount_rate = 0.95

# Exploration Paramaters
decay_rate = 0.00005 # epsilon decay rate per step
min_epsilon = 0.01
max_epsilon = 1.0

checkpoint_path = "C:/Users/Fergus/PycharmProjects/AIGym/AIGymRepo/DoomHall/checkpoint/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# -------------------------------------------------------------------------------------------------------------------

# temporal difference training
def td_train(load_weights=False, timed=False, verbose=False):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'C:/Users/Fergus/PycharmProjects/AIGym/AIGymRepo/DoomHall/logs/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    game, possible_actions = create_environment()
    global_step = 0
    current_state_deque = deque()

    epsilon = 1.0 # Chance of taking a random action
    decay_step = 0

    DQN = DuelDQNetwork(state_size=state_size, num_actions=num_actions, learning_rate=learning_rate)
    # Target model used to determine next state Q values
    target_Model = DuelDQNetwork(state_size=state_size, num_actions=num_actions, learning_rate=learning_rate)

    if load_weights:
        DQN.model.load_weights('C:/Users/Fergus/PycharmProjects/AIGym/AIGymRepo/DoomHall/loadcheckpoint/my_checkpoint')

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
            if timed: step_start = time.time()

            if verbose: print('Frame #: ', global_step+1, end="\t")

            # Select epsilon-greedy action
            if timed: action_start = time.time()
            if np.random.rand() < epsilon:
                action = random.choice(possible_actions)
            else:
                action_index = np.argmax(DQN.model(np.reshape(current_state, [-1, *state_size])), axis=1)[0]
                # print('Network Action')
                action = possible_actions[action_index]

            if timed: print("Action Time: ", time.time() - action_start)
            decay_step += 1
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * decay_step)

            if timed: game_start = time.time()

            # print(action_names[np.argmax(action)])
            # print(action)

            reward = game.make_action(action)
            # print ("\tNew Reward:", reward)

            done = game.is_episode_finished()
            if done:
                # Can't grab state when the episode is done
                blank_img = np.zeros((state_size[0], state_size[1]), dtype=np.int)
                current_state_deque, next_state = stack_frames(current_state_deque, preprocess_frame(blank_img), False)
                # Save SARSA experience
                experience.store((current_state, action, reward, next_state, done))
                step = max_steps

            else:
                # Grab state' after action
                state = game.get_state()
                img = state.screen_buffer
                current_state_deque, next_state = stack_frames(current_state_deque, preprocess_frame(img), False)
                # Save SARSA experience
                experience.store((current_state, action, reward, next_state, done))
                # Update the old state
                current_state = next_state
            if timed: print("Game Time: ", time.time() - game_start)

            if timed: batch_start = time.time()
            # Create sample batches for training
            index_batch, ISWeights_batch, training_batch = experience.sample(batch_size)
            current_state_batch = np.array([each[0] for each in training_batch])
            action_batch = np.array([each[1] for each in training_batch])
            reward_batch = np.array([each[2] for each in training_batch])
            next_state_batch = np.array([each[3] for each in training_batch])
            done_batch = np.array([each[4] for each in training_batch])

            if timed: print("Batch Time: ", time.time() - batch_start)
            if timed: predict_start = time.time()

            # Update target model after certain number of steps
            if global_step % target_update == 0:
                target_Model.model.set_weights(DQN.model.get_weights())
                print("Target Model Updated.")

            # For Double DQN: action is chosen by on-line network, evaluated by target network
            next_Q_action = np.argmax(DQN.model(next_state_batch), axis=1)
            next_Q = np.array(target_Model.model(next_state_batch))
            # Select chosen action value from target model outputs
            # next_Q = next_Q[range(next_Q.shape[0]), next_Q_action]

            # Fill targets with model outputs using current state and only change index of action
            # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
            target_Q_batch = DQN.model(current_state_batch)
            target_Q_batch = np.array(target_Q_batch)
            if verbose: print("Max Output: {:.4f}".format(np.max(target_Q_batch)), end='\t')
            for i in range(len(training_batch)):
                action_sample = np.argmax(action_batch[i])
                if done_batch[i]:
                    target_Q_batch[i][action_sample] = reward_batch[i]
                else:
                    target_Q_batch[i][action_sample] = reward_batch[i] + discount_rate * next_Q[i, next_Q_action[i]]

            # Check for NA inputs and outputs
            if verbose: print("Epsilon: {:.4f}".format(epsilon), end='\t')

            if timed: print("Predict Time: ", time.time() - predict_start)

            # Training on a single batch
            if timed: fit_start = time.time()

            loss, abs_errors = DQN.weighted_fit_gradient(current_state_batch, target_Q_batch, ISWeights_batch)
            experience.update_batch(index_batch, abs_errors)

            # loss = DQN.train_loss.result()
            # DQN.train_loss.reset_states()

            if timed: print("Fit Time: ", time.time() - fit_start)

            if verbose: print('Loss: {:.4f}'.format(loss))

            if timed: save_start = time.time()
            # Check weights on exploding gradients
            if loss > 1000000:
                for i, layer in enumerate(DQN.model.layers):
                    if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
                        w = layer.get_weights()[0]
                        print("Max W{}: {}".format(i, np.max(w)))
                        b = layer.get_weights()[1]
                        print("Max B{}: {}".format(i, np.max(b)))

            # Saving layer weights and bias for Tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('Loss', loss, step=global_step)
                if global_step % 10 == 0:
                    for i, layer in enumerate(DQN.model.layers):
                        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense)):
                            w = layer.get_weights()[0]
                            b = layer.get_weights()[1]
                            tf.summary.histogram("Layer"+str(i)+"Weights", w, step=global_step)
                            tf.summary.histogram("Layer"+str(i)+"Bias", b, step=global_step)
                            tf.summary.scalar("Layer"+str(i)+"WeightsMean", np.mean(w), step=global_step)
                            tf.summary.scalar("Layer"+str(i)+"BiasMean", np.mean(b), step=global_step)
                            tf.summary.scalar("Layer"+str(i)+"WeightsStd", np.std(w), step=global_step)
                            tf.summary.scalar("Layer"+str(i)+"BiasStd", np.std(b), step=global_step)
                            tf.summary.scalar("DQN Prob", epsilon, step=global_step)

            if timed: print("TF Write Time: {}".format(time.time() - save_start))
            if timed: print("Step Time: {}\n".format(time.time() - step_start))

            global_step += 1

        with train_summary_writer.as_default():
            tf.summary.scalar("Episode Reward", game.get_total_reward(), step=global_step)
        print("Episode {} Frame: {} Reward: {} Loss: {:.4f} Epsilon: {:.4f}".format(episode+1,global_step+1,game.get_total_reward(), loss, epsilon))
        episode_rewards.append(game.get_total_reward())

        if episode % episode_save == 0:
            DQN.model.save_weights('C:/Users/Fergus/PycharmProjects/AIGym/AIGymRepo/DoomHall/checkpoint/my_checkpoint')
            print("Model Saved")

    print("Average Episode Reward:", np.mean(episode_rewards))

    game.close()


def test_agent(n):
    game, possible_actions = create_environment()

    DQN = DuelDQNetwork(state_size=state_size, num_actions=game.get_available_buttons_size(), learning_rate=learning_rate)
    DQN.model.load_weights('C:/Users/Fergus/PycharmProjects/AIGym/AIGymRepo/DoomHall/checkpoint/my_checkpoint')

    current_state_deque = deque()

    episode_rewards = []

    max_test_steps = 500

    for i in range(n):

        game.new_episode()
        img = game.get_state().screen_buffer
        current_state_deque, current_state = stack_frames(current_state_deque, preprocess_frame(img), True)

        steps = 0

        while not game.is_episode_finished() and steps < max_test_steps:
            steps += 1

            action_index = np.argmax(DQN.model(np.reshape(current_state, [-1, *state_size])), axis=1)[0]
            action = possible_actions[action_index]

            reward = game.make_action(action)
            print("\tAction: ", action_names[action_index])

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

    DQN = DuelDQNetwork(state_size=state_size, num_actions=num_actions, learning_rate=learning_rate)
    DQN.model.load_weights('C:/Users/Fergus/PycharmProjects/AIGym/AIGymRepo/DoomHall/loadcheckpoint/my_checkpoint')

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

    DQN = DuelDQNetwork(state_size=state_size, num_actions=num_actions, learning_rate=learning_rate)
    # DQN.model.load_weights('C:/Users/Fergus/PycharmProjects/AIGym/AIGymRepo/DoomHall/loadcheckpoint/my_checkpoint')

    loss = 0

    # while loss < 100:
    training_batch = experience.sample(sample_size)
    current_state_batch = np.array([each[0] for each in training_batch])
    action_batch = np.array([each[1] for each in training_batch])
    reward_batch = np.array([each[2] for each in training_batch])
    next_state_batch = np.array([each[3] for each in training_batch])
    done_batch = np.array([each[4] for each in training_batch])

    current_output = DQN.model(current_state_batch)
    print("Current Q: ", current_output)
    print("Action: ", action_batch)
    print("Action Mult: ", tf.multiply(current_output, action_batch))
    print("Reduce Sum: ", tf.reduce_sum(tf.multiply(current_output, action_batch), axis=1))
    current_Q = tf.reduce_sum(tf.multiply(current_output, action_batch), axis=1)
    next_Q = DQN.model(next_state_batch)
    print("Next Q: ", next_Q)
    next_Q_action = np.max(DQN.model.predict(next_state_batch), axis=1)
    print("Next Q Max: ", next_Q_action)

    target_Q_batch = DQN.model(current_state_batch)
    target_Q_batch = np.array(target_Q_batch)
    for i in range(len(training_batch)):
        action_sample = np.argmax(action_batch[i])
        if done_batch[i]:
            target_Q_batch[i][action_sample] = reward_batch[i]
        else:
            target_Q_batch[i][action_sample] = reward_batch[i] + discount_rate * np.max(next_Q[i])

    print("Target Q: ", target_Q_batch)
    print("Test Loss: ", tf.reduce_mean(tf.square(target_Q_batch-current_output)))
    print("Test Loss: ", tf.reduce_mean(tf.reduce_sum(tf.square(target_Q_batch-current_output), axis=1)))
    loss = tf.keras.losses.mean_squared_error(y_true=target_Q_batch, y_pred=current_output)
    print("Loss: ", loss)

def simple_overfit():
    DQN = DuelDQNetwork(state_size=state_size, num_actions=3, learning_rate=learning_rate)

    img = tf.random.uniform((5, *state_size))
    target = np.array([[0, -1, 0], [0,0,100], [0,-6,0], [-1,0,0], [100,0,0]])
    model_output = DQN.model(img)
    print("Init Output: ",model_output)
    print("Target: ", target)

    DQN.model.fit(img, target,epochs=1000, verbose=0)
    fit_model_output = DQN.model(img)
    print("Fit Output: ", fit_model_output)

    print(DQN.model.layers)

if __name__ == "__main__":

    start = time.time()

    pretrain(pretrain_length=pre_train_length)
    td_train(load_weights=False, timed=False)
    # test_agent(n=100)
    # simple_overfit()
    # output_check_batch()
    # test_environment()

    # game, possible_actions = create_environment()
    # DQN = DuelDQNetwork(state_size=state_size, num_actions=num_actions, learning_rate=learning_rate)

    # print("Learning Rate: ", tf.keras.backend.get_value(DQN.model.optimizer.lr))
    # vars = DQN.model.trainable_variables
    # for i,item in enumerate(vars):
    #     print("Vars{} :{}".format(i,item.shape))
    # game.close()

    print("Total Runtime: %s seconds" % (time.time()-start))