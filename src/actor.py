import numpy as np
import random
from abc import ABC, abstractmethod
from omegaconf import DictConfig

from src.utils import random_argmax
from src.critic import MonQCritic


class Actor(ABC):
    def __init__(self, critic: MonQCritic):
        self.critic = critic
        self._train = True
        self.reset()

    @abstractmethod
    def __call__(self, obs_env, obs_mon, t, beta, test=False, rng=np.random):
        """
        Draw one action in one state. Not vectorized.
        """
        pass

    def greedy_call(self, obs_env, obs_mon, t, beta, test=False, rng=np.random):
        """
        Draw the greedy action, i.e., the one maximizing the critic's estimate
        of the state-action value. Not vectorized.
        """
        if not test:
            q_explore = self.critic.obsrv_q[obs_env, obs_mon]
            ae_explore, am_explore = tuple(random_argmax(q_explore, rng))
            q_exploit = self.critic.joint_q[obs_env, obs_mon]
            ae_exploit, am_exploit = tuple(random_argmax(q_exploit, rng))
            if self.critic.joint_count[obs_env, obs_mon, ae_explore, am_explore] > np.log(t) / beta:
                return ae_exploit, am_exploit
            else:
                return ae_explore, am_explore
        else:
            q = self.critic.joint_q[obs_env, obs_mon]
            return tuple(random_argmax(q, rng))
        # q_explore = self.critic.obsrv_q[obs_env, obs_mon]
        # return tuple(random_argmax(q_explore, rng))


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
            eps (DictConfig): configuration to initialize the exploration coefficient
                epsilon,
        """

        Actor.__init__(self, critic)

    def __call__(self, obs_env, obs_mon, t, beta, test=False, rng=np.random):
        return self.greedy_call(obs_env, obs_mon, t, beta, test, rng)

    def update(self):
        pass

    def reset(self):
        pass
