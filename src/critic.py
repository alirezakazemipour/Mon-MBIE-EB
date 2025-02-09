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
        self.beta_e = kwargs["beta_e"]
        self.beta_m = kwargs["beta_m"]
        self.beta = kwargs["beta"]
        self.beta_obs = kwargs["beta_obs"]
        self.beta_kl_ucb = kwargs["beta_kl_ucb"]
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
        self.env_transit_cnt = None
        self.env_term = None
        self.env_obsrv_cnt = None
        self.mon_r = None
        self.joint_cnt = None
        self.joint_obsrv_cnt = None
        self.joint_transit_cnt = None
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
            self.env_obsrv_cnt[obs_env, act_env] += 1
            self.env_r[obs_env, act_env] += rwd_env
            self.joint_obsrv_cnt[obs_env, obs_mon, act_env, act_mon] += 1

        self.env_visit[obs_env, act_env] += 1
        self.env_transit_cnt[obs_env, act_env, next_obs_env] += 1
        self.joint_cnt[obs_env, obs_mon, act_env, act_mon] += 1
        self.mon_r[obs_mon, act_mon] += rwd_mon
        self.joint_transit_cnt[obs_env, obs_mon, act_env, act_mon, next_obs_env, next_obs_mon] += 1

        if term:
            self.env_term[obs_env, act_env] = 1

        return 0

    def opt_pess_mbie(self, rng):  # noqa

        env_rwd = self.update_rwd_model(self.env_obs_space,
                                        self.env_act_space,
                                        self.env_obsrv_cnt,
                                        self.env_rwd_model,
                                        self.beta_e
                                        )

        mon_rwd = self.update_rwd_model(self.mon_obs_space,
                                        self.mon_act_space,
                                        self.joint_cnt.sum((0, 2)),
                                        self.mon_rwd_model,
                                        self.beta_m
                                        )

        ucb4transit = np.zeros_like(self.monitor)
        for s in self.joint_obs_space:
            t = self.joint_cnt[*s].sum()
            f_t = f(t)
            for a in self.joint_act_space:
                if self.joint_cnt[*s, *a] != 0:
                    ucb = self.beta * math.sqrt(math.log(f_t) / self.joint_cnt[*s, *a])
                    ucb4transit[*s, *a] += ucb

        self.joint_q = self.value_iteration(self.vi_iter,
                                            self.joint_q,
                                            self.joint_max_q,
                                            self.joint_cnt.flatten(),
                                            env_rwd[:, None, :, None] + mon_rwd[None, :, None, :] + ucb4transit,
                                            self.gamma,
                                            self.joint_dynamics.reshape(-1, self.env_num_obs * self.mon_num_obs),
                                            jittable_joint_max(self.joint_q),
                                            self.env_term
                                            )

    def obsrv_mbie(self, rng):  # noqa
        obsrv_rwd_bar = np.zeros_like(self.monitor)
        for s in self.joint_obs_space:
            t = self.joint_cnt[*s].sum()
            f_t = f(t)
            for a in self.joint_act_space:
                se, sm = s
                ae, am = a
                if self.joint_cnt[*s, *a] != 0:
                    if self.env_obsrv_cnt[se, ae] == 0:
                        obsrv_rwd_bar[*s, *a] = kl_confidence(t,
                                                              0,
                                                              self.joint_cnt[*s, *a],
                                                              self.beta_kl_ucb
                                                              )
                    # optimism for transitions
                    ucb = self.beta_obs * math.sqrt(math.log(f_t) / self.joint_cnt[*s, *a])
                    obsrv_rwd_bar[*s, *a] += ucb

        self.obsrv_q = self.value_iteration(self.vi_iter,
                                            self.obsrv_q,
                                            1 / (1 - self.gamma),
                                            self.joint_cnt.flatten(),
                                            obsrv_rwd_bar,
                                            self.gamma,
                                            self.joint_dynamics.reshape(-1, self.env_num_obs * self.mon_num_obs),
                                            jittable_joint_max(self.obsrv_q),
                                            np.zeros_like(self.env_term)
                                            )

    def reset(self):
        self.env_r = np.zeros((self.env_num_obs, self.env_num_act))
        self.env_visit = np.zeros((self.env_num_obs, self.env_num_act))
        self.env_transit_cnt = np.zeros((self.env_num_obs, self.env_num_act, self.env_num_obs))
        self.env_term = np.zeros((self.env_num_obs, self.env_num_act))
        self.env_obsrv_cnt = np.zeros((self.env_num_obs, self.env_num_act))
        self.mon_r = np.zeros((self.mon_num_obs, self.mon_num_act))
        self.joint_cnt = np.zeros((self.env_num_obs, self.mon_num_obs, self.env_num_act, self.mon_num_act))
        self.joint_obsrv_cnt = np.zeros((self.env_num_obs, self.mon_num_obs, self.env_num_act, self.mon_num_act))
        self.joint_transit_cnt = np.zeros((self.env_num_obs, self.mon_num_obs, self.env_num_act, self.mon_num_act,
                                           self.env_num_obs, self.mon_num_obs))
        self.joint_q = np.ones(
            (self.env_num_obs, self.mon_num_obs, self.env_num_act, self.mon_num_act)) * self.joint_max_q
        self.obsrv_q = np.ones(
            (self.env_num_obs, self.mon_num_obs, self.env_num_act, self.mon_num_act)) * 1 / (1 - self.gamma)

    @property
    def env_rwd_model(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            r = self.env_r / self.env_obsrv_cnt
        r[np.isnan(r)] = self.env_min_r
        return r

    @property
    def mon_rwd_model(self):
        r = self.mon_r / (self.joint_cnt.sum((0, 2)) + 1e-6)
        return r

    @property
    def joint_dynamics(self):
        p_joint = self.joint_transit_cnt / (self.joint_cnt[..., None, None] + 1e-6)
        return p_joint

    @property
    def monitor(self):
        m = self.joint_obsrv_cnt / (self.joint_cnt + 1e-6)
        return m

    @staticmethod
    @jit
    def update_rwd_model(obs_space, act_space, cnt, rwd_model, a0):
        for s in obs_space:
            t = cnt[s].sum()
            f_t = f(t)
            for a in act_space:
                if cnt[s, a] != 0:
                    ucb = a0 * np.sqrt(np.log(f_t) / cnt[s, a])
                    rwd_model[s, a] += ucb
        return rwd_model

    @staticmethod
    @jit
    def value_iteration(n_iter,
                        q,
                        max_q,
                        cnt,
                        rwd,
                        gamma,
                        p,
                        v,
                        term
                        ):
        """
        Synchronous value iteration
        """
        for _ in range(n_iter):
            z = p @ np.ravel(v).T
            z = z.reshape(rwd.shape)
            q = rwd + gamma * z * (1 - term[:, None, :, None])
            q = q.flatten()
            # q = np.minimum(q, max_q)
            q[cnt == 0] = max_q
            q = q.reshape(rwd.shape)
            v = jittable_joint_max(q)
        return q
