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
                 A=0.1,
                 B=0.1,
                 C=0.1,
                 D=0.1,
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

        self._r_env = None
        self._n_env = None
        self._r_mon = None
        self._n_joint = None
        self._p_joint = None
        self._c_joint = None
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
            self._n_env._table[obs_env, act_env] += 1
            self._r_env._table[obs_env, act_env] += rwd_env
            self._c_joint._table[obs_env, obs_mon, act_env, act_mon] += 1

        self._n_joint._table[obs_env, obs_mon, act_env, act_mon] += 1
        self._r_mon._table[obs_env, obs_mon, act_env, act_mon] += rwd_mon
        self._p_joint._table[obs_env, obs_mon, act_env, act_mon, next_obs_env, next_obs_mon] += 1

        return 0

    def calc_opti_q(self):
        r_env_bar = dc(self._r_env)
        for s in range(self.n_obs_env):
            for a in range(self.n_act_env):
                if self._n_env(s, a) != 0:
                    r_env_bar.update(s,
                                     a,
                                     target=self._r_env(s, a) / self._n_env(s, a) + self.A / math.sqrt(self._n_env(s, a)
                                                                                                       )
                                     )
                else:
                    r_env_bar.update(s, a, target=np.nan)

        r_mon_bar = dc(self._r_mon)
        for se in range(self.n_obs_env):
            for ae in range(self.n_act_env):
                for sm in range(self.n_obs_mon):
                    for am in range(self.n_act_mon):
                        s = se, sm
                        a = ae, am
                        if self._n_joint(*s, *a) != 0:
                            r_mon_bar.update(*s, *a, target=(self._r_mon(*s, *a) / self._n_joint(*s, *a) +
                                                           self.B / math.sqrt(self._n_joint(*s, *a))
                                                           )
                                             )
                        else:
                            r_mon_bar.update(*s, *a, target=np.nan)

        v_joint = np.max(self._q_joint(), axis=(-2, -1))
        s_star = np.unravel_index(np.argmax(v_joint, axis=None), v_joint.shape)

        p_joint_hat = (self._p_joint)
        for se in range(self.n_obs_env):
            for ae in range(self.n_act_env):
                for sm in range(self.n_obs_mon):
                    for am in range(self.n_act_mon):
                        s = se, sm
                        a = ae, am
                        if self._n_joint(*s, *a) != 0:
                            p_joint_hat.update(*s, *a, target=self._p_joint(*s, *a) / self._n_joint(*s, *a))
                        else:
                            p_joint_hat.update(*s, *a, target=np.nan)

                        if self._n_joint(*s, *a) != 0:
                            p_joint_bar = dc(p_joint_hat._table)
                            p_joint_bar[..., *s_star] = np.clip(
                                 p_joint_bar[..., *s_star] + 0.5 * self.C / math.sqrt(self._n_joint(*s, *a)), 0, 1)
                            v_min = np.inf
                            tmp = None
                            for nse in range(self.n_obs_env):
                                for nsm in range(self.n_obs_mon):
                                    ns = nse, nsm
                                    if p_joint_bar[*s, *a, *ns] > 0 and v_joint[*ns] < v_min:
                                        tmp = ns
                                        v_min = v_joint[ns]
                            if tmp is not None:
                                p_joint_bar[..., tmp] = np.clip(p_joint_hat(*s, *a) - 0.5 * self.C / math.sqrt(self._n_joint(*s, *a)),
                                                        0, 1)

        c_joint_bar = dc(self._c_joint)
        for se in range(self.n_obs_env):
            for ae in range(self.n_act_env):
                for sm in range(self.n_obs_mon):
                    for am in range(self.n_act_mon):
                        s = se, sm
                        a = ae, am
                        if self._n_joint(*s, *a) != 0:
                            if self._c_joint(*s, *a) > 0:
                                c_joint_bar.update(*s, *a, target=min(self._c_joint(*s, *a) / self._n_joint(*s, *a) +
                                                                    0.5 * self.D / math.sqrt(self._n_joint(*s, *a)), 1)
                                                   )
                            else:
                                c_joint_bar.update(*s, *a, target=max(self._c_joint(*s, *a) / self._n_joint(*s, *a) -
                                                                    0.5 * self.D / math.sqrt(self._n_joint(*s, *a)), 0)
                                                   )
                        else:
                            c_joint_bar.update(*s, *a, target=np.nan)

        for se in range(self.n_obs_env):
            for ae in range(self.n_act_env):
                if self._n_env(se, ae) == 0:
                    self._q_joint._table[se, :, ae, :] = 1 / (1 - self.gamma)
                    continue
                w = self.A / math.sqrt(self._n_env(se, ae))
                for sm in range(self.n_obs_mon):
                    for am in range(self.n_act_mon):
                        if self._n_joint(se, sm, ae, am) == 0:
                            self._q_joint.update(se, sm, ae, am, target=1 / (1 - self.gamma))
                            continue
                        for nse in range(self.n_obs_env):
                            for nsm in range(self.n_obs_mon):
                                self._q_joint(se, sm, ae, am,
                                              target=r_env_bar(se, ae) + r_mon_bar(se, sm, ae, am) + w * c_joint_bar(se, sm, ae, am) * r_env_bar(se, ae) +
                                                     self._p_joint(se, sm, ae, am).T @
                                                     np.max(self._q_joint(nse, nsm), axis=(-2, -1))
                                              )

        pp = 26

    def reset(self):
        self._r_env.reset()
        self._n_env.reset()
        self._r_mon.reset()
        self._n_joint.reset()
        self._p_joint.reset()
        self._c_joint.reset()
        self._q_joint.reset()


class MonQTableCritic(MonQCritic):
    """
    Instance of MonQCritic that uses tabular Q-function critics.
    """

    def __init__(
            self,
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

        self._r_env = Table(n_obs_env, n_act_env)
        self._n_env = Table(n_obs_env, n_act_env)
        self._r_mon = Table(n_obs_env, n_obs_mon, n_act_env, n_act_mon)
        self._n_joint = Table(n_obs_env, n_obs_mon, n_act_env, n_act_mon)
        self._p_joint = Table(n_obs_env, n_obs_mon, n_act_env, n_act_mon, n_obs_env, n_obs_mon)
        self._c_joint = Table(n_obs_env, n_obs_mon, n_act_env, n_act_mon)
        self._q_joint = Table(n_obs_env, n_obs_mon, n_act_env, n_act_mon, init_value=1 / (1 - self.gamma))

        self.reset()
