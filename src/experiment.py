import math

import gymnasium as gym
import numpy as np
from tqdm import tqdm
import warnings

from src.actor import Actor
from src.critic import MonQTableCritic
from src.utils import set_rng_seed, cantor_pairing


class MonExperiment:
    def __init__(self,
                 env: gym.Env,
                 env_test: gym.Env,
                 actor: Actor,
                 critic: MonQTableCritic,
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

        self._env = env
        self._env_test = env_test

        self._actor = actor
        self._critic = critic
        self._gamma = critic.gamma
        self.beta = kwargs["beta"]

        self._training_steps = training_steps
        self._testing_episodes = testing_episodes
        self._testing_frequency = testing_frequency

        self._rng_seed = rng_seed
        self._hide_progress_bar = hide_progress_bar

    def train(self):
        set_rng_seed(self._rng_seed)
        self._actor.reset()
        self._critic.reset()

        tot_steps = 0
        explore_steps = 0
        tot_episodes = 0
        last_ep_return_env = np.nan
        last_ep_return_mon = np.nan
        test_return_env = np.nan
        test_return_mon = np.nan
        pbar = tqdm(total=self._training_steps, disable=self._hide_progress_bar)

        return_train_history = []
        return_test_history = []
        while tot_steps < self._training_steps:
            pbar.update(tot_steps - pbar.n)
            last_ep_return = last_ep_return_env + last_ep_return_mon
            test_return = test_return_env + test_return_mon
            pbar.set_description(f"train {last_ep_return:.3f} / "
                                 f"test {np.mean(test_return):.3f} "
                                 )
            ep_seed = cantor_pairing(self._rng_seed, tot_episodes)
            rng = np.random.default_rng(ep_seed)
            if math.log(tot_steps + 1e-4) / (explore_steps + 1e-4) > self.beta:
                explore = True
                self._critic.calc_visit_q(rng)
            else:
                explore = False
                self._critic.calc_opti_q(rng)
            obs, _ = self._env.reset(seed=ep_seed)
            ep_return_env = 0.0
            ep_return_mon = 0.0
            ep_steps = 0
            tot_episodes += 1

            while True:
                if tot_steps % self._testing_frequency == 0:
                    self._actor.eval()
                    self._critic.eval()
                    test_return_env, test_return_mon = self.test()
                    self._actor.train()
                    self._critic.train()
                    with warnings.catch_warnings():  # ignore 'mean of empty slice'
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        test_dict = {"test/return_env": test_return_env.mean(),
                                     "test/return_mon": test_return_mon.mean(),
                                     "test/return": (test_return_env + test_return_mon).mean(),
                                     }
                    return_test_history.append(test_dict["test/return"])

                train_dict = {
                    "train/return_env": last_ep_return_env,
                    "train/return_mon": last_ep_return_mon,
                    "train/return": last_ep_return_env + last_ep_return_mon,
                }
                return_train_history.append(train_dict["train/return"])

                tot_steps += 1
                if explore:
                    explore_steps += 1
                act = self._actor(obs["env"], obs["mon"], explore, rng)
                act = {"env": act[0], "mon": act[1]}
                next_obs, rwd, term, trunc, info = self._env.step(act)

                self._critic.update(np.asarray([obs["env"]]),
                                    np.asarray([obs["mon"]]),
                                    np.asarray([act["env"]]),
                                    np.asarray([act["mon"]]),
                                    np.asarray([rwd["env"]]),
                                    np.asarray([rwd["mon"]]),
                                    np.asarray([rwd["proxy"]]),
                                    np.asarray([term]),
                                    np.asarray([next_obs["env"]]),
                                    np.asarray([next_obs["mon"]]),
                                    )
                self._actor.update()

                ep_return_env += (self._gamma ** ep_steps) * rwd["env"]
                ep_return_mon += (self._gamma ** ep_steps) * rwd["mon"]

                ep_steps += 1
                obs = next_obs

                if term or trunc:
                    break

                if tot_steps >= self._training_steps:
                    break

            last_ep_return_env = ep_return_env
            last_ep_return_mon = ep_return_mon

        self._env.close()
        self._env_test.close()
        pbar.close()

        return return_test_history, self._critic.n_tot_env, self._critic.q_joint.mean((-1, -3))

    def test(self):
        ep_return_env = np.zeros(self._testing_episodes)
        ep_return_mon = np.zeros(self._testing_episodes)
        for ep in range(self._testing_episodes):
            ep_seed = cantor_pairing(self._rng_seed, ep)
            obs, _ = self._env_test.reset(seed=ep_seed)
            rng = np.random.default_rng(ep_seed)
            ep_steps = 0
            while True:
                act = self._actor(obs["env"], obs["mon"], False, rng)
                act = {"env": act[0], "mon": act[1]}
                next_obs, rwd, term, trunc, info = self._env_test.step(act)
                ep_return_env[ep] += (self._gamma ** ep_steps) * rwd["env"]
                ep_return_mon[ep] += (self._gamma ** ep_steps) * rwd["mon"]
                if term or trunc:
                    break
                obs = next_obs
                ep_steps += 1

        return ep_return_env, ep_return_mon
