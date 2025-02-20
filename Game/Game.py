from abc import ABC, abstractmethod
import gym

class AbstractGame(gym.Env, ABC):
    @property
    @abstractmethod
    def action_space(self):
        pass
 
    @property
    @abstractmethod
    def observation_space(self):
        pass

    @abstractmethod
    def reset(self, *, seed=None, options=None):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def render(self, mode="human"):
        pass

    @abstractmethod
    def close(self):
        pass
