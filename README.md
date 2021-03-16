# AIGymRepo

This repo contains all the work done in developing and implementing various reinforcement learning techniques on the environments in Doom and Sonic the Hedgehog. This name was selected because it was initially used to test algorithms on the AIGym environments but has evolved to tackle more difficult problems.

The majority of the concepts are taken from Thomas Simonini's Deep RL Course below. 

https://github.com/simoninithomas/Deep_reinforcement_learning_Course

Each directory contains a number of saved models used for testing. They are sub-directories containing checkpoints that can only be accessed using Google's Tensorboard UI. These checkpoints monitor the change in the model's error, how the network parameters change over time, and other custom metrics.

https://www.tensorflow.org/tensorboard

# A2C

This directory contains all implementations of the Actor Critic methods on the Sonic the Hedgehog environment. The goal of Sonic the Hedgehog is to complete the level by reach the goal without getting hit by enemies. This was tested on the AIGym CartPole task before moving on the Sonic game. 

# Doom

Implements the Deep Q-Network approach to solve the Doom target environment. The goal of this environment is for the agent to move to the target and fire at the enemy.

# DoomHall

Implements the Dueling Deep Q-Network approach to solve the Doom corridor environment. The goal of this environment is for the agent to navigate a corridor of enemies and reach the end goal. The primary obstacle is for the agent to learn to defeat the enemies before moving forward.

# PPO 

This directory contains all implementations of the Proximal Policy Optimization methods on the Sonic the Hedgehog environment. Similarly to A2C, this was tested on the AIGym CartPole task before moving on the Sonic game. 

# PolicyGradient 

Implements the Policy Gradients learning approach to the Doom Health environment. The goal of the environment is for the agent to survive as long as possible by finding and grabbing health packs while the agent is continuously losing health. 



