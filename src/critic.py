import numpy as np
import math
from abc import ABC, abstractmethod
from omegaconf import DictConfig
from src.utils import random_argmax, random_argmin, kl_confidence
import itertools

f = lambda t: 1 + t * math.log(t) ** 2


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
        self.a = kwargs["ucb_re"]
        self.b = kwargs["ucb_rm"]
        self.c = kwargs["ucb_p"]

        self.env_num_obs = env_num_obs
        self.mon_num_obs = mon_num_obs
        self.env_num_act = env_num_act
        self.mon_num_act = mon_num_act
        self.joint_obs_space = list(itertools.product(range(self.env_num_obs), range(self.mon_num_obs)))
        self.joint_act_space = list(itertools.product(range(self.env_num_act), range(self.mon_num_act)))
        self.env_obs_space = list(range(self.env_num_obs))
        self.env_act_space = list(range(self.env_num_act))

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
        self.mon_r[obs_env, obs_mon, act_env, act_mon] += rwd_mon
        self.joint_transit_count[obs_env, obs_mon, act_env, act_mon, next_obs_env, next_obs_mon] += 1

        if term:
            self.env_term[obs_env, act_env] = 1

        return 0

    def opt_pess_mbie(self, rng):  # noqa

        env_rwd_model = self.env_rwd_model
        for s in self.env_obs_space:
            for a in self.env_act_space:
                if self.env_visit[s, a] != 0:
                    ucb = self.a / math.sqrt(self.env_visit[s, a])
                    env_rwd_model[s, a] += ucb

        mon_rwd_bar = self.mon_rwd_model
        for s in self.joint_obs_space:
            for a in self.joint_act_space:
                if self.joint_count[*s, *a] != 0:
                    ucb = self.b / math.sqrt(self.joint_count[*s, *a])
                    mon_rwd_bar[*s, *a] += ucb

        p_joint_bar = self.joint_dynamics
        joint_v = np.max(self.joint_q, axis=(-2, -1))
        s_star = random_argmax(joint_v, rng)

        for s in self.joint_obs_space:
            for a in self.joint_act_space:
                if self.joint_count[*s, *a] != 0:
                    ucb = 0.5 * self.c / math.sqrt(self.joint_count[*s, *a])
                    if p_joint_bar[*s, *a, *s_star] + ucb <= 1:
                        p_joint_bar[*s, *a, *s_star] += ucb
                        residual = -ucb
                    else:
                        residual = p_joint_bar[*s, *a, *s_star] - 1
                        p_joint_bar[*s, *a, *s_star] = 1

                    next_states = []
                    for ns in self.joint_obs_space:
                        if p_joint_bar[*s, *a, *ns] > 0 and ns != s_star:
                            next_states.append((ns, joint_v[*ns]))
                    next_states.sort(key=lambda x: x[-1])

                    for ns, _ in next_states:
                        if p_joint_bar[*s, *a, *ns] + residual >= 0:
                            p_joint_bar[*s, *a, *ns] += residual
                            break
                        else:
                            residual = p_joint_bar[*s, *a, *ns] + residual
                            p_joint_bar[*s, *a, *ns] = 0

        for s in self.joint_obs_space:
            for a in self.joint_act_space:
                se, sm = s
                ae, am = a
                if self.env_visit[se, ae] == 0:
                    self.joint_q[se, :, ae, :] = self.joint_max_q
                elif self.joint_count[*s, *a] == 0:
                    self.joint_q[*s, *a] = self.joint_max_q
                else:
                    self.joint_q[*s, *a] = (env_rwd_model[se, ae] + mon_rwd_bar[*s, *a]
                                            + self.gamma * np.ravel(p_joint_bar[*s, *a]).T @ np.ravel(joint_v)
                                            * (1 - self.env_term[se, ae])
                                            )

    def plan4monitor(self, seg, aeg, rng):
        self.obsrv_q = np.zeros_like(self.monitor)
        for s in self.joint_obs_space:
            for a in self.joint_act_space:
                se, sm = s
                ae, am = a
                if (se, ae) == (seg, aeg):
                    t = self.joint_count[*s].sum((-2, -1))
                    if self.joint_count[*s, *a] != 0:
                        self.obsrv_q[*s, *a] = kl_confidence(t, self.monitor[*s, *a], self.joint_count[*s, *a])
                    else:
                        self.obsrv_q[*s, *a] = 1

        smg, amg = random_argmax(self.obsrv_q[seg, :, aeg, :], rng)
        sg_t = seg, smg
        ag_t = aeg, amg

        sgs = [[*sg_t, *ag_t]]
        p_joint = self.joint_dynamics
        expanded = set()
        updated = {(seg, amg, aeg, amg)}

        while len(sgs) > 0:
            seg, smg, aeg, amg = sgs.pop(0)
            sg = seg, smg
            ag = aeg, amg
            if (sg, ag) in expanded:
                continue
            expanded.add((sg, ag))
            predecs = np.argwhere(p_joint[..., *sg] > 0)
            for predec in predecs:
                sgs.append(predec)
                seg, amg, aeg, amg = predec
                if (seg, amg, aeg, amg) in updated:
                    continue
                v_obs = np.max(self.obsrv_q, axis=(-1, -2))
                self.obsrv_q[*predec] = self.gamma * np.ravel(p_joint[*predec]).T @ np.ravel(v_obs)
                # updated.add((seg, amg, aeg, amg))
        return sg_t, ag_t

    def reset(self):
        self.env_r = np.zeros((self.env_num_obs, self.env_num_act))
        self.env_visit = np.zeros((self.env_num_obs, self.env_num_act))
        self.env_term = np.zeros((self.env_num_obs, self.env_num_act))
        self.env_obsrv_count = np.zeros((self.env_num_obs, self.env_num_act))
        self.mon_r = np.zeros((self.env_num_obs, self.mon_num_obs, self.env_num_act, self.mon_num_act))
        self.joint_count = np.zeros((self.env_num_obs, self.mon_num_obs, self.env_num_act, self.mon_num_act))
        self.joint_obsrv_count = np.zeros((self.env_num_obs, self.mon_num_obs, self.env_num_act, self.mon_num_act))
        self.joint_transit_count = np.zeros((self.env_num_obs, self.mon_num_obs, self.env_num_act, self.mon_num_act,
                                             self.env_num_obs, self.mon_num_obs))
        self.joint_q = np.ones(
            (self.env_num_obs, self.mon_num_obs, self.env_num_act, self.mon_num_act)) * self.joint_max_q

    @property
    def env_rwd_model(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            r = self.env_r / self.env_obsrv_count
        r[np.isnan(r)] = self.env_min_r
        return r

    @property
    def mon_rwd_model(self):
        r = self.mon_r / (self.joint_count + 1e-4)
        return r

    @property
    def joint_dynamics(self):
        p_joint = self.joint_transit_count / (self.joint_count[..., None, None] + 1e-4)
        return p_joint

    @property
    def joint_num_obs(self):
        return self.env_num_obs * self.mon_num_obs

    @property
    def monitor(self):
        m = self.joint_obsrv_count / (self.joint_count + 1e-4)
        return m
