import tensorflow as tf
import numpy as np

def WeightWriter(tf_writer, model, layer_tuple, global_step):
    with tf_writer.as_default():
        for i, layer in enumerate(model.model.layers):
            if isinstance(layer, layer_tuple):
                w = layer.get_weights()[0]
                b = layer.get_weights()[1]
                tf.summary.histogram("Layer" + str(i) + "Weights", w, step=global_step)
                tf.summary.histogram("Layer" + str(i) + "Bias", b, step=global_step)
                tf.summary.scalar("Layer" + str(i) + "WeightsMean", np.mean(w), step=global_step)
                tf.summary.scalar("Layer" + str(i) + "BiasMean", np.mean(b), step=global_step)
                tf.summary.scalar("Layer" + str(i) + "WeightsStd", np.std(w), step=global_step)
                tf.summary.scalar("Layer" + str(i) + "BiasStd", np.std(b), step=global_step)


def TestRewardWriter(summary_writer, envfn, agent, n_episodes, global_step):

    env = envfn()
    state_size = env.observation_space.shape

    obs = env.reset()

    total_reward_list = []

    steps = 0
    max_steps = 2048

    for i_episode in range(n_episodes):
        done = False
        total_reward = 0
        while not done and steps < max_steps:
            obs = np.reshape(obs, (-1, *state_size))
            action_probs = np.array(agent.model(obs)[0])
            sel_action = np.argmax(action_probs)
            obs, rew, done, info = env.step(sel_action)
            # env.render()
            total_reward += rew
            steps += 1
        total_reward_list.append(total_reward)
        obs = env.reset()
        steps = 0

    total_reward_list = np.array(total_reward_list)

    sum_reward = np.sum(total_reward_list)
    avg_reward = np.mean(total_reward_list)
    max_reward = np.max(total_reward_list)
    std_reward = np.std(total_reward_list)
    with summary_writer.as_default():
        tf.summary.scalar("MaxReward", max_reward, step=global_step)
        tf.summary.scalar("AverageReward", avg_reward, step=global_step)
        tf.summary.scalar("StdReward", std_reward, step=global_step)

    return total_reward_list