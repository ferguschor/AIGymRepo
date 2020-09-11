import numpy as np
import SonicEnv
import cv2

# Class taking multiple Python / Gym Environments and runs actions in sequence
class MultiEnv:

    # Input a list of callable functions which create environments
    def __init__(self, env_fns):
        # Creates the list of envs
        self.envs = [fn() for fn in env_fns]
        self.n_envs = len(env_fns)

        env = self.envs[0]
        obs_space = env.observation_space

        self.buf_obs = np.zeros((self.n_envs,) + (obs_space.shape), dtype=obs_space.dtype)
        self.buf_rews = np.zeros((self.n_envs,), dtype=np.float32)
        self.buf_dones = np.zeros((self.n_envs,), dtype=np.bool)
        self.buf_infos = [{} for _ in range(self.n_envs)]
        self.actions = None

    # Take a step in each environment based on the list of actions given
    # Returns the array of step outputs
    def step(self, actions):
        assert len(actions) == self.n_envs, (len(actions), self.n_envs)

        for i, e in enumerate(self.envs):
            self.buf_obs[i], self.buf_rews[i], self.buf_dones[i], self.buf_infos[i] = e.step(actions[i])
            if self.buf_dones[i]:
                self.buf_obs[i] = e.reset() # Reset if the episode is done, can save because return is s'

        return np.copy(self.buf_obs), np.copy(self.buf_rews), np.copy(self.buf_dones), np.copy(self.buf_infos)

    # Reset all environments and update observation buffer
    def reset(self):
        for i, e in enumerate(self.envs):
            self.buf_obs[i] = e.reset()
        return np.copy(self.buf_obs)

def main():
    # tmp = MultiEnv([env.make_env_4])
    # obs = tmp.reset()
    # obs = obs[0]
    # print(obs)
    env = MultiEnv([SonicEnv.make_env_3] * 2)
    obs = env.reset()
    while True:
        action = env.envs[0].action_space.sample()
        obs, rew, done, info = env.step([action])
        env.envs[0].render()
        obs = np.squeeze(obs[0])
        print(obs.shape)
        obs = obs / 255.
        for i in range(4):
            print(np.mean(obs, axis=(0,1)))
            cv2.imshow('GrayScale'+str(i), np.squeeze(obs[:,:,i]))
        if done:
            obs = env.reset()

if __name__ == "__main__":
    main()
