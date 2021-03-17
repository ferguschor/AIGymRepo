# AIGymRepo

This repo contains all the work done in developing and implementing various reinforcement learning techniques on the environments in Doom and Sonic the Hedgehog. This name was selected because it was initially used to test algorithms on the AIGym environments but has evolved to tackle more difficult problems.

The majority of the concepts are built on top of Thomas Simonini's Deep RL Course below. 

https://github.com/simoninithomas/Deep_reinforcement_learning_Course

Each directory contains a number of saved models used for testing. They are sub-directories containing checkpoints that can only be accessed using Google's Tensorboard UI. These checkpoints monitor the change in the model's error, how the network parameters change over time, and other custom metrics.

https://www.tensorflow.org/tensorboard

# A2C

This directory contains all implementations of the Actor Critic methods on the Sonic the Hedgehog environment. The goal of Sonic the Hedgehog is to complete the level by reach the goal without getting hit by enemies. This was tested on the AIGym CartPole task before moving on the Sonic game. 

The base Actor-Critic method consists of two networks, one to represent the Actor and the second to represent the Critic. The Actor is a network that receives the screen pixels as input and outputs a probability distribution for which actions to take. The Critic network also receives the screen as input and is used to determine the current "state value". This Critic method is very similar to the Deep Q-Network model. These networks are typically combined as one and only differ in the final layers.

# Doom

Implements the Deep Q-Network approach to solve the Doom target environment. The goal of this environment is for the agent to move to the target and fire at the enemy. 

The Deep Q-Network approach is based on calculating the "state-action value" or expected reward of the current state and selecting an action. As the agent experiments with random actions in the environment, it will eventually reach a terminal state (victory or death). By discounting the reward at this terminal state, a state value can be approximated for the previous states. This is repeated until the network sufficiently converges. 

# DoomHall

Implements the Dueling Deep Q-Network approach to solve the Doom corridor environment. The goal of this environment is for the agent to navigate a corridor of enemies and reach the end goal. The primary obstacle is for the agent to learn to defeat the enemies before moving forward.

The Dueling Deep Q-network makes use of two Deep Q-Networks to improve the value function. One network is primarily used to value the state. The second network calculates the "advantage" or the difference between the true value of the action-state and the expected value. The second network provides insight into the advantage of selecting one action over the others.

# PPO 

This directory contains all implementations of the Proximal Policy Optimization methods on the Sonic the Hedgehog environment. Similarly to A2C, this was tested on the AIGym CartPole task before moving on the Sonic game. 

The PPO approach is based on the Policy Gradient in that the model is trying to learn the optimal action policy. The main difference between PPO and VPG is the step method in the model learning. One of the shortfalls of VPG is that it can remain in a local optima and never explore to find a better policy. PPO aims to resolve this by ensuring the new policy does not stray too far from the old policy while sufficiently improving its performance. 

# PolicyGradient 

Implements the Policy Gradients learning approach to the Doom Health environment. The goal of the environment is for the agent to survive as long as possible by finding and grabbing health packs while the agent is continuously losing health. 

The Vanilla Policy Gradient approach makes use of an action policy or a probability distribution of actions. As the model is trained, actions that garner more rewards have their probabilities increased and less desirable actions have their probabilities decreased. Although valuing the state is still used in the network, it is less considered in the agent's actions like in a DQN. 



