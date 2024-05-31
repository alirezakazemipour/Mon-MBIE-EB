import numpy as np
import math
from abc import ABC, abstractmethod
from omegaconf import DictConfig
from copy import deepcopy as dc
import src.parameter as parameter
from src.approximator import Table


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
                 A=0.0004,
                 B=0.0004,
                 C=0.0001,
                 D=0.0004,
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
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.n_obs_env = None
        self.n_obs_mon = None
        self.n_act_env = None
        self.n_act_mon = None

        self._nr_env = None
        self._nd_env = None
        self._n_env = None
        self._nr_mon = None
        self._n_joint = None
        self._np_joint = None
        self._nc_joint = None
        self._q_joint = None

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
            self._n_env[obs_env, act_env] += 1
            self._nr_env[obs_env, act_env] += rwd_env
            self._nc_joint[obs_env, obs_mon, act_env, act_mon] += 1

        self._n_joint[obs_env, obs_mon, act_env, act_mon] += 1
        self._nr_mon[obs_env, obs_mon, act_env, act_mon] += rwd_mon
        self._np_joint[obs_env, obs_mon, act_env, act_mon, next_obs_env, next_obs_mon] += 1

        if term:
            self._nd_env[obs_env, act_env] = 1

        return 0

    def calc_opti_q(self, ):
        r_env_bar = np.ones((self.n_obs_env, self.n_act_env))
        for s in range(self.n_obs_env):
            for a in range(self.n_act_env):
                if self._n_env[s, a] != 0:
                    r_env_bar[s, a] = (self._nr_env[s, a] / self._n_env[s, a] +
                                       math.sqrt(math.log(self._n_env[s].sum(-1))) * self.A / math.sqrt(
                                self._n_env[s, a]))

        r_mon_bar = np.zeros((self.n_obs_env, self.n_obs_mon, self.n_act_env, self.n_act_mon))
        for se in range(self.n_obs_env):
            for ae in range(self.n_act_env):
                for sm in range(self.n_obs_mon):
                    for am in range(self.n_act_mon):
                        s = se, sm
                        a = ae, am
                        if self._n_joint[*s, *a] != 0:
                            r_mon_bar[*s, *a] = (self._nr_mon[*s, *a] / self._n_joint[*s, *a] +
                                                 math.sqrt(
                                                     math.log(self._n_joint[*s].sum((-1, -2)))) * self.B / math.sqrt(
                                        self._n_joint[*s, *a])
                                                 )

        p_joint_hat = np.ones((self.n_obs_env, self.n_obs_mon, self.n_act_env,
                               self.n_act_mon, self.n_obs_env, self.n_obs_mon)
                              ) / self.n_obs_env / self.n_obs_mon
        for se in range(self.n_obs_env):
            for ae in range(self.n_act_env):
                for sm in range(self.n_obs_mon):
                    for am in range(self.n_act_mon):
                        s = se, sm
                        a = ae, am
                        if self._n_joint[*s, *a] != 0:
                            p_joint_hat[*s, *a] = self._np_joint[*s, *a] / self._n_joint[*s, *a]

        v_joint = np.max(self._q_joint, axis=(-2, -1))
        s_star = np.unravel_index(np.argmax(v_joint, axis=None), v_joint.shape)
        for se in range(self.n_obs_env):
            for ae in range(self.n_act_env):
                for sm in range(self.n_obs_mon):
                    for am in range(self.n_act_mon):
                        s = se, sm
                        a = ae, am
                        if self._n_joint[*s, *a] != 0:
                            if p_joint_hat[*s, *a, *s_star] + math.sqrt(
                                    math.log(self._n_joint[*s].sum((-1, -2)))) * 0.5 * self.C / math.sqrt(
                                self._n_joint[*s, *a]) <= 1:
                                p_joint_hat[*s, *a, *s_star] += math.sqrt(
                                    math.log(self._n_joint[*s].sum((-1, -2)))) * 0.5 * self.C / math.sqrt(
                                    self._n_joint[*s, *a])
                                residual = -0.5 * math.sqrt(
                                    math.log(self._n_joint[*s].sum((-1, -2)))) * self.C / math.sqrt(
                                    self._n_joint[*s, *a])
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

        c_joint_bar = np.ones((self.n_obs_env, self.n_obs_mon, self.n_act_env, self.n_act_mon)) / 2
        for se in range(self.n_obs_env):
            for ae in range(self.n_act_env):
                for sm in range(self.n_obs_mon):
                    for am in range(self.n_act_mon):
                        s = se, sm
                        a = ae, am
                        if self._n_joint[*s, *a] != 0:
                            c_joint_bar[*s, *a] = np.clip(self._nc_joint[*s, *a] / self._n_joint[*s, *a] +
                                                          math.sqrt(
                                                              math.log(self._n_joint[*s].sum(
                                                                  (-1, -2)))) * self.D / math.sqrt(
                                self._n_joint[*s, *a])
                                                          , 0,
                                                          1
                                                          )

        for se in range(self.n_obs_env):
            for ae in range(self.n_act_env):
                if np.sum(c_joint_bar[se, :, ae, :]) < 0.01:
                    self._q_joint[se, :, ae, :] = -2 / (1 - self.gamma)
                    continue
                if self._n_env[se, ae] == 0:
                    self._q_joint[se, :, ae, :] = 30#2 / (1 - self.gamma)
                    continue
                w = math.sqrt(math.log(self._n_env[se].sum(-1))) * self.A / math.sqrt(self._n_env[se, ae])
                for sm in range(self.n_obs_mon):
                    for am in range(self.n_act_mon):
                        s = se, sm
                        a = ae, am
                        if self._n_joint[*s, *a] == 0:
                            self._q_joint[*s, *a] = 30#2 / (1 - self.gamma)
                            continue
                        else:
                            self._q_joint[*s, *a] = (r_env_bar[se, ae] + r_mon_bar[*s, *a] + w * c_joint_bar[*s, *a]
                                                     + self.gamma * np.ravel(p_joint_hat[*s, *a]).T @ np.ravel(v_joint)
                                                     * (1 - self._nd_env[se, ae])
                                                     )

    def reset(self):
        self._nr_env = np.zeros((self.n_obs_env, self.n_act_env))
        self._nd_env = np.zeros((self.n_obs_env, self.n_act_env))
        self._n_env = np.zeros((self.n_obs_env, self.n_act_env))
        self._nr_mon = np.zeros((self.n_obs_env, self.n_obs_mon, self.n_act_env, self.n_act_mon))
        self._n_joint = np.zeros((self.n_obs_env, self.n_obs_mon, self.n_act_env, self.n_act_mon))
        self._np_joint = np.zeros((self.n_obs_env, self.n_obs_mon, self.n_act_env, self.n_act_mon,
                                   self.n_obs_env, self.n_obs_mon)
                                  )
        self._nc_joint = np.zeros((self.n_obs_env, self.n_obs_mon, self.n_act_env, self.n_act_mon))
        self._q_joint = np.ones(
            (self.n_obs_env, self.n_obs_mon, self.n_act_env, self.n_act_mon)) * 30#2 / (1 - self.gamma)


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

        self._nr_env = np.zeros((n_obs_env, n_act_env))
        self._nd_env = np.zeros((n_obs_env, n_act_env))
        self._n_env = np.zeros((n_obs_env, n_act_env))
        self._nr_mon = np.zeros((n_obs_env, n_obs_mon, n_act_env, n_act_mon))
        self._n_joint = np.zeros((n_obs_env, n_obs_mon, n_act_env, n_act_mon))
        self._np_joint = np.zeros((n_obs_env, n_obs_mon, n_act_env, n_act_mon, n_obs_env, n_obs_mon))
        self._nc_joint = np.zeros((n_obs_env, n_obs_mon, n_act_env, n_act_mon))
        self._q_joint = np.ones((n_obs_env, n_obs_mon, n_act_env, n_act_mon)) * 1 / (1 - self.gamma)

        self.reset()
