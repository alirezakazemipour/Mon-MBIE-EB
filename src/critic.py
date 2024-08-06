import numpy as np
import math
from abc import ABC, abstractmethod
from omegaconf import DictConfig
from src.utils import random_argmax

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
        self.q_max = kwargs["q_max"]
        self.r_min = kwargs["r_min"]
        self.A = kwargs["ucb_re"]
        self.B = kwargs["ucb_rm"]
        self.C = kwargs["ucb_p"]

        self.n_obs_env = None
        self.n_obs_mon = None
        self.n_act_env = None
        self.n_act_mon = None

        self.nr_env = None
        self.nd_env = None
        self.n_env = None
        self.nr_mon = None
        self.n_joint = None
        self.np_joint = None
        self.np_env = None
        self.q_joint = None
        self.n_tot_env = None
        self.np_env = None
        self.q_visit = None
        self.rwd_model = None

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
        if not np.isnan(rwd_proxy)[0]:
            self.n_env[obs_env, act_env] += 1
            self.nr_env[obs_env, act_env] += rwd_env
            self.rwd_model[obs_env, act_env] = self.nr_env[obs_env, act_env] / self.n_env[obs_env, act_env]

        self.n_tot_env[obs_env, act_env] += 1
        self.np_env[obs_env, act_env, next_obs_env] += 1
        self.n_joint[obs_env, obs_mon, act_env, act_mon] += 1
        self.nr_mon[obs_env, obs_mon, act_env, act_mon] += rwd_mon
        self.np_joint[obs_env, obs_mon, act_env, act_mon, next_obs_env, next_obs_mon] += 1

        if term:
            self.nd_env[obs_env, act_env] = 1

        return 0

    def calc_opti_q(self, rng):

        r_mon_bar = np.zeros((self.n_obs_env, self.n_obs_mon, self.n_act_env, self.n_act_mon))
        for se in range(self.n_obs_env):
            for ae in range(self.n_act_env):
                for sm in range(self.n_obs_mon):
                    for am in range(self.n_act_mon):
                        s = se, sm
                        a = ae, am
                        if self.n_joint[*s, *a] != 0:
                            t = self.n_joint[*s].sum((-2, -1))
                            f_t = f(t)
                            ucb = self.B * math.sqrt(math.log(f_t) / self.n_joint[*s, *a])
                            r_mon_bar[*s, *a] = self.nr_mon[*s, *a] / self.n_joint[*s, *a] + ucb

        p_joint_hat = np.ones((self.n_obs_env, self.n_obs_mon, self.n_act_env,
                               self.n_act_mon, self.n_obs_env, self.n_obs_mon)
                              ) / self.n_obs_env / self.n_obs_mon
        for se in range(self.n_obs_env):
            for ae in range(self.n_act_env):
                for sm in range(self.n_obs_mon):
                    for am in range(self.n_act_mon):
                        s = se, sm
                        a = ae, am
                        if self.n_joint[*s, *a] != 0:
                            p_joint_hat[*s, *a] = self.np_joint[*s, *a] / self.n_joint[*s, *a]

        v_joint = np.max(self.q_joint, axis=(-2, -1))
        s_star = random_argmax(v_joint, rng)
        for se in range(self.n_obs_env):
            for ae in range(self.n_act_env):
                for sm in range(self.n_obs_mon):
                    for am in range(self.n_act_mon):
                        s = se, sm
                        a = ae, am
                        if self.n_joint[*s, *a] != 0:
                            ucb = 0.5 * self.C * math.sqrt(1 / self.n_joint[*s, *a])
                            if p_joint_hat[*s, *a, *s_star] + ucb <= 1:
                                p_joint_hat[*s, *a, *s_star] += ucb
                                residual = -ucb
                            else:
                                residual = p_joint_hat[*s, *a, *s_star] - 1
                                p_joint_hat[*s, *a, *s_star] = 1

                            next_states = []
                            for nse in range(self.n_obs_env):
                                for nsm in range(self.n_obs_mon):
                                    ns = nse, nsm
                                    if p_joint_hat[*s, *a, *ns] > 0 and ns != s_star:
                                        next_states.append((ns, v_joint[*ns]))
                            next_states.sort(key=lambda x: x[-1])
                            next_states.reverse()

                            for ns, _ in next_states:
                                if p_joint_hat[*s, *a, *ns] + residual >= 0:
                                    p_joint_hat[*s, *a, *ns] += residual
                                    break
                                else:
                                    residual = p_joint_hat[*s, *a, *ns] + residual
                                    p_joint_hat[*s, *a, *ns] = 0

        for se in range(self.n_obs_env):
            for ae in range(self.n_act_env):
                for sm in range(self.n_obs_mon):
                    for am in range(self.n_act_mon):
                        s = se, sm
                        a = ae, am
                        if self.n_joint[*s, *a] == 0:
                            self.q_joint[*s, *a] = self.q_max
                        else:
                            self.q_joint[*s, *a] = (self.rwd_model[se, ae] + r_mon_bar[*s, *a]
                                                    + self.gamma * np.ravel(p_joint_hat[*s, *a]).T @ np.ravel(
                                        v_joint)
                                                    * (1 - self.nd_env[se, ae])
                                                    )

    def calc_visit_q(self, rng):

        r_obs_bar = np.zeros((self.n_obs_env, self.n_act_env))
        for s in range(self.n_obs_env):
            for a in range(self.n_act_env):
                if self.n_tot_env[s, a] != 0:
                    t = self.n_tot_env[s].sum()
                    f_t = f(t)
                    ucb = self.A * math.sqrt(math.log(f_t) / self.n_tot_env[s, a])
                    r_obs_bar[s, a] = self.n_env[s, a] / self.n_tot_env[s, a] + ucb

        p_env_hat = np.ones((self.n_obs_env, self.n_act_env, self.n_obs_env)) / self.n_obs_env
        for s in range(self.n_obs_env):
            for a in range(self.n_act_env):
                if self.n_tot_env[s, a] != 0:
                    p_env_hat[s, a] = self.np_env[s, a] / self.n_tot_env[s, a]

        v_obs = np.max(self.q_visit, axis=(-1))
        s_star = rng.choice(np.flatnonzero(v_obs == v_obs.max()))
        for s in range(self.n_obs_env):
            for a in range(self.n_act_env):
                if self.n_tot_env[s, a] != 0:
                    ucb = 0.5 * self.C * math.sqrt(1 / self.n_tot_env[s, a])
                    if p_env_hat[s, a, s_star] + ucb <= 1:
                        p_env_hat[s, a, s_star] += ucb
                        residual = -ucb
                    else:
                        residual = p_env_hat[s, a, s_star] - 1
                        p_env_hat[s, a, s_star] = 1
                    next_states = []
                    for ns in range(self.n_obs_env):
                        if p_env_hat[s, a, ns] > 0 and ns != s_star:
                            next_states.append((ns, v_obs[ns]))
                    next_states.sort(key=lambda x: x[-1])
                    next_states.reverse()

                    for ns, _ in next_states:
                        if p_env_hat[s, a, ns] + residual >= 0:
                            p_env_hat[s, a, ns] += residual
                            break
                        else:
                            residual = p_env_hat[s, a, ns] + residual
                            p_env_hat[*s, *a, *ns] = 0

        for s in range(self.n_obs_env):
            for a in range(self.n_act_env):
                if self.n_tot_env[s, a] == 0:
                    self.q_visit[s, a] = 1
                else:
                    tmp = np.sign(self.n_env[s, a])
                    term = np.logical_or(self.nd_env[s, a], tmp)
                    self.q_visit[s, a] = (
                                r_obs_bar[s, a] + self.gamma * np.ravel(p_env_hat[s, a]).T @ np.ravel(v_obs) * (
                                    1 - term)
                                )

    def reset(self):
        self.np_env = np.zeros((self.n_obs_env, self.n_act_env, self.n_obs_env))
        self.rwd_model = np.ones((self.n_obs_env, self.n_act_env)) * self.r_min
        self.nr_env = np.zeros((self.n_obs_env, self.n_act_env))
        self.n_tot_env = np.zeros((self.n_obs_env, self.n_act_env))
        self.nd_env = np.zeros((self.n_obs_env, self.n_act_env))
        self.n_env = np.zeros((self.n_obs_env, self.n_act_env))
        self.nr_mon = np.zeros((self.n_obs_env, self.n_obs_mon, self.n_act_env, self.n_act_mon))
        self.n_joint = np.zeros((self.n_obs_env, self.n_obs_mon, self.n_act_env, self.n_act_mon))
        self.np_joint = np.zeros((self.n_obs_env, self.n_obs_mon, self.n_act_env, self.n_act_mon,
                                  self.n_obs_env, self.n_obs_mon)
                                 )
        self.q_joint = np.ones(
            (self.n_obs_env, self.n_obs_mon, self.n_act_env, self.n_act_mon)) * self.q_max
        self.q_visit = np.ones((self.n_obs_env, self.n_act_env))


class MonQTableCritic(MonQCritic):
    """
    Instance of MonQCritic that uses tabular Q-function critics.
    """

    def __init__(self,
                 n_obs_env: int,
                 n_obs_mon: int,
                 n_act_env: int,
                 n_act_mon: int,
                 **kwargs,
                 ):
        MonQCritic.__init__(self, **kwargs)
        self.n_obs_env = n_obs_env
        self.n_obs_mon = n_obs_mon
        self.n_act_env = n_act_env
        self.n_act_mon = n_act_mon
        self.action_shape = (n_act_env, n_act_mon)

        self.rwd_model = np.zeros((self.n_obs_env, self.n_act_env))
        self.nr_env = np.zeros((n_obs_env, n_act_env))
        self.n_tot_env = np.zeros((n_obs_env, n_act_env))
        self.nd_env = np.zeros((n_obs_env, n_act_env))
        self.n_env = np.zeros((n_obs_env, n_act_env))
        self.nr_mon = np.zeros((n_obs_env, n_obs_mon, n_act_env, n_act_mon))
        self.n_joint = np.zeros((n_obs_env, n_obs_mon, n_act_env, n_act_mon))
        self.np_joint = np.zeros((n_obs_env, n_obs_mon, n_act_env, n_act_mon, n_obs_env, n_obs_mon))
        self.np_env = np.zeros((n_obs_env, n_act_env, n_obs_env))
        self.q_joint = np.ones((n_obs_env, n_obs_mon, n_act_env, n_act_mon)) * self.q_max
        self.q_visit = np.ones((n_obs_env, n_act_env)) * self.q_max

        self.reset()
