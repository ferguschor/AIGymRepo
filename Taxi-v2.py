import numpy as np
import gym
import random

env = gym.make("Taxi-v2")

num_actions = env.action_space.n
num_states = env.observation_space.n
print(num_actions)
print(num_states)

Q = np.zeros((num_states, num_actions))
print (Q)

max_episodes = 20000
max_steps = 99

# learning rate
learning_rate = 0.8
# discount rate
gamma = 0.95

# epsilon for random action
epsilon = 1.0
min_epsilon = 0.01
max_epsilon = 1.0
decay_rate = 0.005

reward_history = []


for i_episode in range(max_episodes):
    # Reset environment
    state = env.reset()
    total_reward = 0

    for i_steps in range(max_steps):
        epsilon_randomizer = random.uniform(0,1)

        # Select random action
        if epsilon_randomizer < epsilon:
            action = env.action_space.sample()
        # Otherwize select best action
        else:
            action = np.argmax(Q[state, :])

        new_state, reward, done, info = env.step(action)

        total_reward += reward

        Q[state, action] = Q[state, action] + learning_rate*(reward + gamma * np.amax(Q[new_state,:]) - Q[state, action])

        state = new_state

        # Reach end goal
        if done:
            break

    # Update epsilon to lower rate for exploitation-exploration-tradeoff
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*i_episode)

    # Log accumulated reward
    reward_history.append(total_reward)

print("Average Acc Reward: %.2f" % (sum(reward_history)/len(reward_history)))
print(Q)

# Play using finalized table

test_results = []
for i_episode in range(1000):
    # Reset environment
    state = env.reset()

    for i_steps in range(max_steps):

        # env.render()

        action = np.argmax(Q[state, :])

        new_state, reward, done, info = env.step(action)

        state = new_state

        # Reach end goal
        if done:
            # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
            # env.render()

            # We print the number of step it took.
            # print("Number of steps", i_steps)
            test_results.append(i_steps)
            break

print("Average steps to completion: %.2f" % (sum(test_results)/len(test_results)))
env.close()
