import numpy as np
import math
from abc import ABC, abstractmethod
from omegaconf import DictConfig
from src.utils import kl_confidence, jittable_joint_max
import itertools
from numba import jit
import time


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
                 mon_rwd_model: np.ndarray,
                 mon_dynamics: np.ndarray,
                 monitor: np.ndarray,
                 **kwargs,
                 ):
        """
        Args:
            strategy (str): can be either "oracle", "reward_model", "q_sequential",
                "q_joint", "ignore", "reward_is_X", where X is a float.
            gamma (float): discount factor,
            lr (DictConfig): configuration to initialize the learning rate,
        """

        self.gamma = kwargs["gamma"]
        self.joint_max_q = kwargs["joint_max_q"]
        self.env_min_r = kwargs["env_min_r"]
        self.a = kwargs["ucb_re"]
        self.b = kwargs["ucb_rm"]
        self.vi_iter = kwargs["vi_iter"]

        self.env_num_obs = env_num_obs
        self.mon_num_obs = mon_num_obs
        self.env_num_act = env_num_act
        self.mon_num_act = mon_num_act
        self.joint_obs_space = list(itertools.product(range(self.env_num_obs), range(self.mon_num_obs)))
        self.joint_act_space = list(itertools.product(range(self.env_num_act), range(self.mon_num_act)))
        self.env_obs_space = list(range(self.env_num_obs))
        self.env_act_space = list(range(self.env_num_act))
        self.mon_rwd_model = mon_rwd_model
        self.monitor = monitor
        self.mon_dynamics = mon_dynamics

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
        self.env_transit_count = None

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
        self.mon_r[obs_env, obs_mon, act_env, act_mon] += rwd_mon
        self.joint_transit_count[obs_env, obs_mon, act_env, act_mon, next_obs_env, next_obs_mon] += 1
        self.env_transit_count[obs_env, act_env, next_obs_env] += 1

        if term:
            self.env_term[obs_env, act_env] = 1

        return 0

    def opt_pess_mbie(self, rng):  # noqa

        env_rwd_model = self.update_env_rwd_model(self.env_obs_space,
                                                  self.env_act_space,
                                                  self.env_obsrv_count,
                                                  self.env_rwd_model,
                                                  self.a
                                                  )

        p_joint_bar = self.env_dynamics[:, None, :, None, :, None] * np.expand_dims(self.mon_dynamics, axis=-2)
        joint_v = np.max(self.joint_q, axis=(-2, -1))

        self.joint_q = self.value_iteration(self.vi_iter,
                                            self.joint_obs_space,
                                            self.joint_act_space,
                                            self.env_visit,
                                            self.joint_q,
                                            self.joint_max_q,
                                            env_rwd_model,
                                            self.mon_rwd_model,
                                            self.gamma,
                                            p_joint_bar,
                                            joint_v,
                                            self.env_term
                                            )

    def obsrv_mbie(self, rng):  # noqa
        env_obsrv_rwd_bar = self.update_env_rwd_model(self.env_obs_space,
                                                      self.env_act_space,
                                                      self.env_visit,
                                                      np.zeros_like(self.env_rwd_model),
                                                      self.b
                                                      )

        mon_obsrv_rwd_bar = self.update_obsrv_rwd_model(self.joint_obs_space,
                                                        self.joint_act_space,
                                                        self.env_obsrv_count,
                                                        np.zeros_like(self.monitor),
                                                        self.monitor
                                                        )

        p_joint_bar = self.env_dynamics[:, None, :, None, :, None] * np.expand_dims(self.mon_dynamics, axis=-2)
        obsrv_v = np.max(self.obsrv_q, axis=(-2, -1))

        self.obsrv_q = self.value_iteration(self.vi_iter,
                                            self.joint_obs_space,
                                            self.joint_act_space,
                                            self.env_visit,
                                            np.zeros_like(self.obsrv_q),
                                            1 / (1 - self.gamma),
                                            env_obsrv_rwd_bar,
                                            mon_obsrv_rwd_bar,  # can be set to 0
                                            self.gamma,
                                            p_joint_bar,
                                            np.zeros_like(obsrv_v),
                                            np.zeros_like(self.env_term)
                                            )

    def reset(self):
        self.env_r = np.zeros((self.env_num_obs, self.env_num_act))
        self.env_visit = np.zeros((self.env_num_obs, self.env_num_act))
        self.env_term = np.zeros((self.env_num_obs, self.env_num_act))
        self.env_obsrv_count = np.zeros((self.env_num_obs, self.env_num_act))
        self.env_transit_count = np.zeros((self.env_num_obs, self.env_num_act, self.env_num_obs,))
        self.mon_r = np.zeros((self.env_num_obs, self.mon_num_obs, self.env_num_act, self.mon_num_act))
        self.joint_count = np.zeros((self.env_num_obs, self.mon_num_obs, self.env_num_act, self.mon_num_act))
        self.joint_obsrv_count = np.zeros((self.env_num_obs, self.mon_num_obs, self.env_num_act, self.mon_num_act))
        self.joint_transit_count = np.zeros((self.env_num_obs, self.mon_num_obs, self.env_num_act, self.mon_num_act,
                                             self.env_num_obs, self.mon_num_obs))
        self.joint_q = np.ones(
            (self.env_num_obs, self.mon_num_obs, self.env_num_act, self.mon_num_act)) * self.joint_max_q
        self.obsrv_q = np.ones(
            (self.env_num_obs, self.mon_num_obs, self.env_num_act, self.mon_num_act)) * self.joint_max_q

    @property
    def env_rwd_model(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            r = self.env_r / self.env_obsrv_count
        r[np.isnan(r)] = self.env_min_r
        return r

    @property
    def env_obsrv_rwd_model(self):
        r = self.env_r / (self.env_obsrv_count + 1e-4)
        return r

    @property
    def env_dynamics(self):
        p_env = self.env_transit_count / (self.env_visit[..., None] + 1e-4)
        return p_env

    @staticmethod
    @jit
    def update_env_rwd_model(env_obs_space, env_act_space, count, env_rwd_model, a0):
        for s in env_obs_space:
            for a in env_act_space:
                if count[s, a] != 0:
                    t = count[s].sum()
                    f_t = f(t)
                    ucb = a0 * np.sqrt(2 * np.log(f_t) / count[s, a])
                    env_rwd_model[s, a] += ucb
        return env_rwd_model

    @staticmethod
    @jit
    def update_obsrv_rwd_model(obs_space, act_space, count, model, monitor):
        for s in obs_space:
            for a in act_space:
                se, sm = s
                ae, am = a
                if count[se, ae] == 0 and np.sum(model[se, :, ae, :]) > 0:
                    model[*s, *a] = monitor[*s, *a]
        return model

    @staticmethod
    @jit
    def value_iteration(num_iter,
                        obs_space,
                        act_space,
                        env_visit,
                        q,
                        max_q,
                        env_rwd,
                        mon_rwd,
                        gamma,
                        p,
                        v,
                        term
                        ):
        for _ in range(num_iter):
            for s in obs_space:
                for a in act_space:
                    se, sm = s
                    ae, am = a
                    if env_visit[se, ae] == 0:
                        q[se, :, ae, :] = max_q
                    else:
                        q[*s, *a] = (env_rwd[se, ae] + mon_rwd[*s, *a] + gamma * np.ravel(p[*s, *a]).T @ np.ravel(v)
                                     * (1 - term[se, ae])
                                     )
                    v = jittable_joint_max(q)
        return q
