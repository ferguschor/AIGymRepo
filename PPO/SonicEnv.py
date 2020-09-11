import retro
import cv2
import numpy as np
import gym
import time
import retrowrapper
from collections import deque
from baselines.common.vec_env import DummyVecEnv
from baselines.common.vec_env import SubprocVecEnv

class PreprocessFrames(gym.ObservationWrapper):

    def __init__(self, env, height, width):
        super(PreprocessFrames, self).__init__(env)

        self.height = height
        self.width = width

        self.shape = (self.width, self.height)

        # correct attributes for observation_space
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.shape, dtype=np.float32)

    def observation(self, observation):
        # Resize screen window to specified H, W
        observation = cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)

        # cv2.Color only takes f32 as type
        # Change screen to gray scale
        observation = observation.astype('float32')
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        observation = observation / 255.

        return observation


class DiscreteActions(gym.ActionWrapper):

    def __init__(self, env):
        super(DiscreteActions, self).__init__(env)
        # Current Buttons
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        # Limit space to only useful actions
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self.actions = []

        # This loop converts the specified actions to a list of True/False to be used as actions in the original env
        for _action in actions:
            tmp = np.array([False] * 12)
            for i in _action:
                tmp[buttons.index(i)] = True
            self.actions.append(tmp)
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action):
        assert (0 <= action < len(self.actions)), action
        return self.actions[action].copy()


class RewardScaler(gym.RewardWrapper):

    def reward(self, reward):
        # Scale reward to add stabilize policy model training
        return reward * 0.01


class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_X = 0
        self._max_X = 0

    def reset(self, **kwargs):
        self._cur_X = 0
        self._max_X = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._cur_X += rew
        # Update reward, do not use max(0, rew) or else AI will just farm rew going back and forth
        # This encourages AI to move forward and reach new ground
        rew = max(0, self._cur_X - self._max_X)
        # Update _max_X
        self._max_X = max(self._cur_X, self._max_X)
        return obs, rew, done, info


class CustomFrameStack(gym.Wrapper):

    def __init__(self, env, maxlen=4):
        super(CustomFrameStack, self).__init__(env)

        self.maxlen = maxlen

        # Deque to hold past frames
        self.frames = deque(maxlen=maxlen)

        # New Lows need to be in the same shape as the previous
        shape = [*self.env.observation_space.shape, maxlen]
        type = self.env.observation_space.dtype

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=type)

    def __get_observation(self):
        assert len(self.frames) == self.maxlen, (len(self.frames), self.maxlen)
        return np.stack(list(self.frames), axis=2)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        [self.frames.append(obs) for _ in range(self.maxlen)]
        return self.__get_observation()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.frames.append(obs)
        return self.__get_observation(), rew, done, info

def make_env(env_idx):
    # list of dictionaries for different levels
    dicts = [
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'SpringYardZone.Act3'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'SpringYardZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'GreenHillZone.Act3'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'GreenHillZone.Act1'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'StarLightZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'StarLightZone.Act1'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'MarbleZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'MarbleZone.Act1'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'MarbleZone.Act3'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'ScrapBrainZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'LabyrinthZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'LabyrinthZone.Act1'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'LabyrinthZone.Act3'}
    ]
    # retrowrapper.set_retro_make(retro.make)
    # env = retrowrapper.RetroWrapper(game=dicts[env_idx]['game'], state=dicts[env_idx]['state'])
    env = retro.make(game=dicts[env_idx]['game'], state=dicts[env_idx]['state'])

    env = DiscreteActions(env)
    # env = RewardScaler(env)
    env = PreprocessFrames(env, 96, 96)
    env = CustomFrameStack(env, maxlen=4)
    env = AllowBacktracking(env)

    return env

# Different functions are created for each field to be used in MultiEnv
# MultiEnv takes in a list of callable functions
def make_env_0():
    return make_env(0)

def make_env_1():
    return make_env(1)

def make_env_2():
    return make_env(2)

def make_env_3():
    return make_env(3)

def make_env_4():
    return make_env(4)

def make_env_5():
    return make_env(5)

def make_env_6():
    return make_env(6)

def make_env_7():
    return make_env(7)

def make_env_8():
    return make_env(8)

def make_env_9():
    return make_env(9)

def make_env_10():
    return make_env(10)

def make_env_11():
    return make_env(11)

def make_env_12():
    return make_env(12)

def main():
    # Alter reward in scenario.json (C:\Users\Fergus\Anaconda3\envs\AIGym\Lib\site-packages\retro\data\stable\SonicTheHedgehog-Genesis)

    env = SubprocVecEnv([make_env_3])
    obs = env.reset()
    # env = make_env_3()
    # env2 = make_env_4()
    print(env.observation_space)
    print(env.action_space.n)
    print(obs.shape)
    print(obs[0].shape)
    # obs = env2.reset()
    rew_mb = []
    dones_mb = []
    obs_mb = []
    step = 0
    while True:
        action = env.action_space.sample()
        obs, rew, done, info = env.step([0])
        print("Step {} Reward: {}, Done: {}".format(step, rew, done))
        rew_mb.append(rew)
        dones_mb.append(done)
        obs_mb.append(obs)
        env.render()

        step += 1
        # obs = obs[1] / 255.
        # for i in range(4):
        #     cv2.imshow('GrayScale'+str(i), np.squeeze(obs[:,:,i]))
        #     cv2.waitKey(1)
        if done[0]:
            env.close()
            break
    rew_mb = np.array(rew_mb)
    dones_mb = np.array(dones_mb)
    obs_mb = np.array(obs_mb)
    print("Rewards: ", rew_mb)
    print(rew_mb.shape)
    print(dones_mb)
    print(dones_mb.shape)
    print(obs_mb.shape)




if __name__ == "__main__":
    main()