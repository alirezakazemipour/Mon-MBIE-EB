import gymnasium
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import numpy as np
import math

from src.utils import dict_to_id
from src.actor import MonEpsilonGreedy
from src.critic import MonQTableCritic
from src.experiment import MonExperiment
from src.wrappers import monitor_wrappers
import matplotlib.pyplot as plt


@hydra.main(version_base=None, config_path="configs", config_name="default")
def run(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    # pprint(config)

    group = dict_to_id(cfg.environment) + "/" + dict_to_id(cfg.monitor)
    wandb.init(
        group=group,
        config=config,
        settings=wandb.Settings(
            start_method="thread",
            _disable_stats=True,
            _disable_meta=True,
        ),
        **cfg.wandb,
    )

    if cfg.monitor.id in [
        "StatelessBinaryMonitor",
        "LimitedUseMonitor",
        "LimitedUseBonusMonitor",
    ]:  # these are fully deterministic monitors
        cfg.experiment.testing_episodes = 1
    if cfg.monitor.id in ["ToySwitchMonitor"]:  # on/off initial monitor state
        cfg.experiment.testing_episodes = 2

    env = gymnasium.make(**cfg.environment)
    cfg.environment.id = cfg.environment.id.replace("-Stochastic", "")  # test with determ. rewards
    # cfg.environment.max_episode_steps = 10  # greedy policies need less than 10 steps to go to goal
    env_test = gymnasium.make(**cfg.environment)
    env = getattr(monitor_wrappers, cfg.monitor.id)(env, **cfg.monitor)
    env_test = getattr(monitor_wrappers, cfg.monitor.id)(env_test, test=True, **cfg.monitor)

    critic = MonQTableCritic(
        env.observation_space["env"].n,
        env.observation_space["mon"].n,
        env.action_space["env"].n,
        env.action_space["mon"].n,
        **cfg.agent.critic,
    )
    actor = MonEpsilonGreedy(critic)
    experiment = MonExperiment(env, env_test, actor, critic, **cfg.experiment)

    return_train_history, return_test_history = experiment.train()
    experiment.test()

    if cfg.experiment.debugdir is not None:
        from plot_gridworld_agent import plot_agent

        savepath = os.path.join(
            cfg.experiment.debugdir,
            group,
            cfg.agent.critic.q0,
        )
        os.makedirs(savepath, exist_ok=True)
        plot_agent(critic, env, savepath)

    if cfg.experiment.datadir is not None:
        filepath = os.path.join(
            cfg.experiment.datadir,
            group,
            cfg.agent.critic.q0,
        )
        os.makedirs(filepath, exist_ok=True)
        strat = cfg.agent.critic.strategy
        seed = str(cfg.experiment.rng_seed)
        savepath = os.path.join(filepath, strat + "_train_" + seed)
        np.save(savepath, np.array(return_train_history))
        savepath = os.path.join(filepath, strat + "_test_" + seed)
        np.save(savepath, np.array(return_test_history))

    wandb.finish()


if __name__ == "__main__":
    run()
    # algos = ["OFU_Solvable_NoPenalty", "OFU_Unsolvable_Cautious", "OFU_Solvable_Penalty"]
    # for algo in algos:
    #     runs = []
    #     for i in range(100):
    #         x = np.load(
    #             f"data/{algo}/iStatelessBinaryMonitor/OFU/reward_model_test_{i}.npy")
    #         runs.append(x)
    #     smoothed = []
    #     for run in runs:
    #         val = [run[0] if not np.isnan(run[0]) else np.nanmin(run)]
    #         for tmp in run[1:]:
    #             tmp = tmp if not np.isnan(tmp) else np.nanmin(run[1:])
    #             val.append(0.9 * val[-1] + 0.1 * tmp)
    #         smoothed.append(val)
    #     mean_return = np.mean(np.asarray(smoothed), axis=0)
    #     std_return = np.std(np.asarray(smoothed), axis=0)
    #     lower_bound = mean_return - 1.96 * std_return / math.sqrt(100)
    #     upper_bound = mean_return + 1.96 * std_return / math.sqrt(100)
    #     plt.fill_between(np.arange(4000),
    #                      lower_bound,
    #                      upper_bound,
    #                      alpha=0.25
    #                      )
    #     plt.plot(np.arange(4000),
    #              mean_return,
    #              alpha=1,
    #              label=algo,
    #              linewidth=3
    #              )
    # plt.axhline(0.951, linestyle='--', label="optimal-Penalty", c="magenta")
    # plt.axhline(0.99, linestyle='--', label="optimal-NoPenalty", c="olive")
    # plt.xlabel("every 10 training steps")
    # plt.ylabel("discounted (test?) return")
    # plt.title(f" performance over {100} runs")
    # plt.grid()
    # plt.legend()
    # plt.ylim([0.925, 0.96])
    # plt.xlim([425, 625])
    # # plt.ylim([-0.1, 1.1])
    #
    # plt.show()
    #
