import numpy as np
from abc import ABC, abstractmethod

from src.utils import random_argmax
from src.critic import MonQCritic


class Actor(ABC):
    def __init__(self, critic: MonQCritic):
        self.critic = critic
        self._train = True
        self.reset()

    @abstractmethod
    def __call__(self, obs_env, obs_mon, explore=False, rng=np.random):
        """
        Draw one action in one state. Not vectorized.
        """
        pass

    def greedy_call(self, obs_env, obs_mon, explore=False, rng=np.random):
        """
        Draw the greedy action, i.e., the one maximizing the critic's estimate
        of the state-action value. Not vectorized.
        """
        if explore:
            q = self.critic.obsrv_q[obs_env, obs_mon]
        else:
            q = self.critic.joint_q[obs_env, obs_mon]
        return random_argmax(q, rng)

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    def eval(self):
        self._train = False

    def train(self):
        self._train = True


class Greedy(Actor):
    def __init__(
            self,
            critic: MonQCritic,
            **kwargs,
    ):
        """
        Args:
            critic (Critic): the critic providing estimates of state-action values,
                epsilon,
        """

        Actor.__init__(self, critic)

    def __call__(self, obs_env, obs_mon, explore=False, rng=np.random):
        return self.greedy_call(obs_env, obs_mon, explore, rng)

    def update(self):
        pass

    def reset(self):
        pass
