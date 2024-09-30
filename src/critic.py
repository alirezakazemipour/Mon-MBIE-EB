import numpy as np
import math
from abc import ABC, abstractmethod
from omegaconf import DictConfig
from src.utils import kl_confidence, jittable_joint_max
import itertools
from numba import jit


@jit
def f(t):
    return 1 + t * np.log(t) ** 2


class Critic(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def update(self, **kwargs):
        pass

    @abstractmethod
    def reset(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass


class MonQCritic(Critic):
    """
    Generic class for Mon-MDP critics.
    Compared to classic critics, MonQCritic has multiple Q-functions, and a
    self._strategy attribute to define update rules and how to treat NaN rewards.
    """

    def __init__(self,
                 env_num_obs: int,
                 mon_num_obs: int,
                 env_num_act: int,
                 mon_num_act: int,
                 gamma: float,
                 **kwargs,
                 ):
        """
        Args:
            strategy (str): can be either "oracle", "reward_model", "q_sequential",
                "q_joint", "ignore", "reward_is_X", where X is a float.
            gamma (float): discount factor,
            lr (DictConfig): configuration to initialize the learning rate,
        """

        self.gamma = gamma
        self.joint_max_q = kwargs["joint_max_q"]
        self.env_min_r = kwargs["env_min_r"]
        self.a = kwargs["ucb_a"]
        self.b = kwargs["ucb_b"]
        self.c = kwargs["ucb_c"]
        self.vi_iter = kwargs["vi_iter"]

        self.env_num_obs = env_num_obs
        self.mon_num_obs = mon_num_obs
        self.env_num_act = env_num_act
        self.mon_num_act = mon_num_act
        self.joint_obs_space = list(itertools.product(range(self.env_num_obs), range(self.mon_num_obs)))
        self.joint_act_space = list(itertools.product(range(self.env_num_act), range(self.mon_num_act)))
        self.env_obs_space = list(range(self.env_num_obs))
        self.env_act_space = list(range(self.env_num_act))
        self.mon_obs_space = list(range(self.mon_num_obs))
        self.mon_act_space = list(range(self.mon_num_act))

        self.env_r = None
        self.env_visit = None
        self.env_term = None
        self.env_obsrv_count = None
        self.mon_r = None
        self.joint_count = None
        self.joint_obsrv_count = None
        self.joint_transit_count = None
        self.joint_q = None
        self.obsrv_q = None

    def update(self,
               obs_env,
               obs_mon,
               act_env,
               act_mon,
               rwd_env,
               rwd_mon,
               rwd_proxy,
               term,
               next_obs_env,
               next_obs_mon,
               ):
        if not np.isnan(rwd_proxy):
            self.env_obsrv_count[obs_env, act_env] += 1
            self.env_r[obs_env, act_env] += rwd_env
            self.joint_obsrv_count[obs_env, obs_mon, act_env, act_mon] += 1

        self.env_visit[obs_env, act_env] += 1
        self.joint_count[obs_env, obs_mon, act_env, act_mon] += 1
        self.mon_r[obs_mon, act_mon] += rwd_mon
        self.joint_transit_count[obs_env, obs_mon, act_env, act_mon, next_obs_env, next_obs_mon] += 1

        if term:
            self.env_term[obs_env, act_env] = 1

        return 0

    def opt_pess_mbie(self, rng):  # noqa

        env_rwd_model = self.update_rwd_model(self.env_obs_space,
                                              self.env_act_space,
                                              self.env_obsrv_count,
                                              self.env_rwd_model,
                                              self.a
                                              )

        mon_rwd_bar = self.update_rwd_model(self.mon_obs_space,
                                            self.mon_act_space,
                                            self.joint_count.sum((0, 2)),
                                            self.mon_rwd_model,
                                            self.b
                                            )

        opt4transit = np.zeros_like(self.monitor)
        for s in self.joint_obs_space:
            for a in self.joint_act_space:
                if self.joint_count[*s, *a] != 0:
                    t = self.joint_count[*s].sum()
                    f_t = f(t)
                    ucb = self.c * math.sqrt(2 * math.log(f_t) / self.joint_count[*s, *a])
                    opt4transit[*s, *a] += ucb

        self.joint_q = self.value_iteration(self.vi_iter,
                                            self.joint_obs_space,
                                            self.joint_act_space,
                                            np.zeros_like(self.joint_q),
                                            self.joint_max_q,
                                            self.joint_count,
                                            env_rwd_model[:, None, :, None] + mon_rwd_bar[None, :, None, :] +
                                            opt4transit,
                                            self.gamma,
                                            self.joint_dynamics,
                                            np.zeros((self.env_num_obs, self.mon_num_obs)),
                                            self.env_term
                                            )

    def obsrv_mbie(self, rng):  # noqa
        # env_obsrv_rwd_bar = self.update_env_obsrv_rwd_model(self.env_obsrv_rwd_model,
        #                                                     self.env_obs_space,
        #                                                     self.env_act_space,
        #                                                     self.env_visit,
        #                                                     self.env_obsrv_count,
        #                                                     )

        # the following reward is more conservative than the base code because it checks all the joint state-action
        # pairs but is theoretically stronger that might not be obvious in our environments.
        mon_obsrv_rwd_bar = np.zeros_like(self.monitor)
        for s in self.joint_obs_space:
            for a in self.joint_act_space:
                se, sm = s
                ae, am = a
                if self.joint_count[*s, *a] != 0:
                    t = self.joint_count[*s].sum()
                    if self.env_obsrv_count[se, ae] == 0:
                        mon_obsrv_rwd_bar[*s, *a] = kl_confidence(t,
                                                                  0,
                                                                  self.joint_count[*s, *a]
                                                                  )
                    # optimism for transitions
                    f_t = f(t)
                    ucb = self.c * math.sqrt(2 * math.log(f_t) / self.joint_count[*s, *a])
                    mon_obsrv_rwd_bar[*s, *a] += ucb

        self.obsrv_q = self.value_iteration(self.vi_iter,
                                            self.joint_obs_space,
                                            self.joint_act_space,
                                            np.zeros_like(self.obsrv_q),
                                            1 / (1 - self.gamma),
                                            self.joint_count,
                                            mon_obsrv_rwd_bar,
                                            self.gamma,
                                            self.joint_dynamics,
                                            np.zeros((self.env_num_obs, self.mon_num_obs)),
                                            np.zeros_like(self.env_term)  # Discuss with Mike to conclude
                                            )

    def reset(self):
        self.env_r = np.zeros((self.env_num_obs, self.env_num_act))
        self.env_visit = np.zeros((self.env_num_obs, self.env_num_act))
        self.env_term = np.zeros((self.env_num_obs, self.env_num_act))
        self.env_obsrv_count = np.zeros((self.env_num_obs, self.env_num_act))
        self.mon_r = np.zeros((self.mon_num_obs, self.mon_num_act))
        self.joint_count = np.zeros((self.env_num_obs, self.mon_num_obs, self.env_num_act, self.mon_num_act))
        self.joint_obsrv_count = np.zeros((self.env_num_obs, self.mon_num_obs, self.env_num_act, self.mon_num_act))
        self.joint_transit_count = np.zeros((self.env_num_obs, self.mon_num_obs, self.env_num_act, self.mon_num_act,
                                             self.env_num_obs, self.mon_num_obs))
        self.joint_q = np.ones(
            (self.env_num_obs, self.mon_num_obs, self.env_num_act, self.mon_num_act)) * self.joint_max_q
        self.obsrv_q = np.ones(
            (self.env_num_obs, self.mon_num_obs, self.env_num_act, self.mon_num_act)) * 1 / (1  - self.gamma)

    @property
    def env_rwd_model(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            r = self.env_r / self.env_obsrv_count
        r[np.isnan(r)] = self.env_min_r
        return r

    @property
    def mon_rwd_model(self):
        r = self.mon_r / (self.joint_count.sum((0, 2)) + 1e-4)
        return r

    @property
    def joint_dynamics(self):
        p_joint = self.joint_transit_count / (self.joint_count[..., None, None] + 1e-4)
        return p_joint

    @property
    def monitor(self):
        m = self.joint_obsrv_count / (self.joint_count + 1e-4)
        return m

    @staticmethod
    @jit
    def update_rwd_model(obs_space, act_space, count, rwd_model, a0):
        for s in obs_space:
            for a in act_space:
                if count[s, a] != 0:
                    t = count[s].sum()
                    f_t = f(t)
                    ucb = a0 * np.sqrt(2 * np.log(f_t) / count[s, a])
                    rwd_model[s, a] += ucb
        return rwd_model

    @staticmethod
    @jit
    def value_iteration(n_iter,
                        obs_space,
                        act_space,
                        q,
                        max_q,
                        count,
                        rwd,
                        gamma,
                        p,
                        v,
                        term
                        ):
        for _ in range(n_iter):
            for s in obs_space:
                for a in act_space:
                    se, sm = s
                    ae, am = a
                    if count[*s, *a] == 0:
                        q[*s, *a] = max_q
                    else:
                        q[*s, *a] = rwd[*s, *a] + gamma * np.ravel(p[*s, *a]).T @ np.ravel(v) * (1 - term[se, ae])
                    v = jittable_joint_max(q)
        return q
