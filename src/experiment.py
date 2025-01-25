import math

import gymnasium as gym
import numpy as np
from tqdm import tqdm
import warnings

from src.actor import Actor
from src.critic import MonQCritic
from src.utils import set_rng_seed, cantor_pairing
from src.wrappers.monitor_wrappers import Button


class MonExperiment:
    def __init__(self,
                 env: gym.Env,
                 env_test: gym.Env,
                 actor: Actor,
                 critic: MonQCritic,
                 training_steps: int,
                 testing_episodes: int,
                 testing_frequency: int,
                 rng_seed: int = 1,
                 hide_progress_bar: bool = True,
                 **kwargs,
                 ):
        """
        Args:
            env (gymnasium.Env): environment used to collect training samples,
            env_test (gymnasium.Env): environment used to test the greedy policy,
            actor (Actor): actor to draw actions,
            critic (Critic): critic to evaluate state-action pairs,
            training_steps (int): how many environment steps training will last,
            testing_episodes (int): number of episodes to test the greedy policy,
            testing_frequency (int): after how many training steps the greedy
                policy will be tested,
            rng_seed (int): to fix random seeds for reproducibility,
            hide_progress_bar (bool): to show tqdm progress bar with some basic info,
        """

        self.env = env
        self.env_test = env_test

        self.actor = actor
        self.critic = critic
        self.gamma = critic.gamma
        self.beta = kwargs["beta"]

        self.training_steps = training_steps
        self.testing_episodes = testing_episodes
        self.testing_frequency = testing_frequency

        self.rng_seed = rng_seed
        self.hide_progress_bar = hide_progress_bar
        self.tot_episodes = None
        self.explore_episodes = None

    def train(self):
        set_rng_seed(self.rng_seed)
        self.actor.reset()
        self.critic.reset()

        tot_steps = 0
        self.tot_episodes = 0
        self.explore_episodes = 0
        last_ep_return_env = np.nan
        last_ep_return_mon = np.nan
        test_return_env = np.nan
        test_return_mon = np.nan
        pbar = tqdm(total=self.training_steps, disable=self.hide_progress_bar)

        return_train_history = []
        return_test_history = []
        goal_cnt_hist = []
        button_cnt_hist = []
        unobsrv_cnt_hist = []
        while tot_steps < self.training_steps:
            pbar.update(tot_steps - pbar.n)
            last_ep_return = last_ep_return_env + last_ep_return_mon
            test_return = test_return_env + test_return_mon
            pbar.set_description(f"train {last_ep_return:.3f} / "
                                 f"test {np.mean(test_return):.3f} "
                                 )
            ep_seed = cantor_pairing(self.rng_seed, self.tot_episodes)
            rng = np.random.default_rng(ep_seed)

            self.critic.opt_pess_mbie(rng)  # off-policy; can be updated every episode!
            explore = False
            ################
            if math.log(self.tot_episodes + 1e-4, self.beta) > self.explore_episodes:
                explore = True
                self.explore_episodes += 1
                self.critic.obsrv_mbie(rng)
            ################

            obs, _ = self.env.reset(seed=ep_seed)
            ep_return_env = 0.0
            ep_return_mon = 0.0
            ep_steps = 0
            self.tot_episodes += 1

            while True:
                if tot_steps % self.testing_frequency == 0:
                    self.actor.eval()
                    self.critic.eval()
                    test_return_env, test_return_mon = self.test()
                    self.actor.train()
                    self.critic.train()
                    with warnings.catch_warnings():  # ignore 'mean of empty slice'
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        test_dict = {"test/return_env": test_return_env.mean(),
                                     "test/return_mon": test_return_mon.mean(),
                                     "test/return": (test_return_env + test_return_mon).mean(),
                                     }
                    return_test_history.append(test_dict["test/return"])
                    if self.env.spec.id == gym.envs.spec("Gym-Grid/Gridworld-Snake-6x6-v0").id and isinstance(self.env,
                                                                                                              Button):
                        goal_cnt_hist.append(self.critic.env_visit[-1, 4])
                        button_cnt_hist.append(self.critic.env_visit[31, 1])
                        unobsrv_cnt_hist.append(self.critic.env_visit[[2, 8, 20, 26, 32]].mean())

                train_dict = {
                    "train/return_env": last_ep_return_env,
                    "train/return_mon": last_ep_return_mon,
                    "train/return": last_ep_return_env + last_ep_return_mon,
                }
                return_train_history.append(train_dict["train/return"])

                tot_steps += 1
                act = self.actor(obs["env"], obs["mon"], explore, rng)
                act = {"env": act[0], "mon": act[1]}
                next_obs, rwd, term, trunc, info = self.env.step(act)

                self.critic.update(obs["env"],
                                   obs["mon"],
                                   act["env"],
                                   act["mon"],
                                   rwd["env"],
                                   rwd["mon"],
                                   rwd["proxy"],
                                   term,
                                   next_obs["env"],
                                   next_obs["mon"],
                                   )

                ep_return_env += (self.gamma ** ep_steps) * rwd["env"]
                ep_return_mon += (self.gamma ** ep_steps) * rwd["mon"]

                ep_steps += 1
                obs = next_obs

                if term or trunc:
                    break

                if tot_steps >= self.training_steps:
                    break

            last_ep_return_env = ep_return_env
            last_ep_return_mon = ep_return_mon

        self.env.close()
        self.env_test.close()
        pbar.close()

        data = {"test_return": return_test_history,
                "env_visit": self.critic.env_visit,
                "joint_q": self.critic.joint_q,
                "obsrv_q": self.critic.obsrv_q,
                "joint_count": self.critic.joint_count,
                "joint_obsrv_count": self.critic.joint_obsrv_count,
                "monitor": self.critic.monitor,
                "env_obsrv_count": self.critic.env_obsrv_count,
                "env_reward_model": self.critic.env_rwd_model,
                "goal_cnt_hist": goal_cnt_hist,
                "button_cnt_hist": button_cnt_hist,
                "unobsrv_cnt_hist": unobsrv_cnt_hist
                }
        return data

    def test(self):
        ep_return_env = np.zeros(self.testing_episodes)
        ep_return_mon = np.zeros(self.testing_episodes)
        for ep in range(self.testing_episodes):
            ep_seed = cantor_pairing(self.rng_seed, ep)
            obs, _ = self.env_test.reset(seed=ep_seed)
            rng = np.random.default_rng(ep_seed)
            ep_steps = 0
            while True:
                act = self.actor(obs["env"], obs["mon"], False, rng)
                act = {"env": act[0], "mon": act[1]}
                next_obs, rwd, term, trunc, info = self.env_test.step(act)
                ep_return_env[ep] += (self.gamma ** ep_steps) * rwd["env"]
                ep_return_mon[ep] += (self.gamma ** ep_steps) * rwd["mon"]
                if term or trunc:
                    break
                obs = next_obs
                ep_steps += 1

        return ep_return_env, ep_return_mon
