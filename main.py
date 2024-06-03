import gymnasium
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import numpy as np
import math
from tqdm import tqdm
from src.utils import dict_to_id
from src.actor import MonEpsilonGreedy
from src.critic import MonQTableCritic
from src.experiment import MonExperiment
from src.wrappers import monitor_wrappers
import matplotlib.pyplot as plt


@hydra.main(version_base=None, config_path="configs", config_name="default")
def run(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

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

    env = gymnasium.make(**cfg.environment)
    env_test = gymnasium.make(**cfg.environment)
    env = getattr(monitor_wrappers, cfg.monitor.id)(env, **cfg.monitor)
    env_test = getattr(monitor_wrappers, cfg.monitor.id)(env_test, test=True, **cfg.monitor)

    # ret = []
    # for i in tqdm(range(10000)):
    #     np.random.seed(i)
    #     ret_e = 0
    #     obs, _ = env.reset(seed=i)
    #     t = 0
    #     while True:
    #         while obs["mon"] == 1:
    #             a = {"env": 0, "mon": 0}
    #             obs, r, term, trunc, _ = env.step(a)
    #             ret_e += (0.99 ** t) * (r["env"] + r["mon"])
    #             t += 1
    #
    #         a = {"env": 1, "mon": 0}
    #         obs, r, term, trunc, _ = env.step(a)
    #         ret_e += (0.99 ** t) * (r["env"] + r["mon"])
    #         if term or trunc:
    #             ret.append(ret_e)
    #             break
    #         t += 1
    #
    # print(np.mean(ret))
    # print(np.std(ret))
    # exit()

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
    # run()
    # exit()
    # algos = ["Alireza", "Simone", "Eps"]
    algos = ["iGym-Monitor/Gridworld-Penalty-3x3-v0_mes50_rmNone/iButtonMonitor/OFU"]
    for algo in algos:
        runs = []
        for i in range(100, 200):
            if algo == "iGym-Monitor/Gridworld-Penalty-3x3-v0_mes50_rmNone/iButtonMonitor/OFU":
                x = np.load(
                    f"data/{algo}/reward_model_test_{i}.npy")
            elif algo == "Simone":
                x = np.load(
                    f"data/{algo}/q_visit_-10.0_0.0_1.0_0.01_{i}.npz")["test/return"]
            else:
                x = np.load(
                    f"data/{algo}/eps_greedy_-10.0_0.0_1.0_0.01_{i}.npz")["test/return"]
            runs.append(x)
        # print(np.argmin(np.nansum(np.asarray(runs), axis=-1)))
        # exit()
        smoothed = []
        for run in runs:
            val = [run[0] if not np.isnan(run[0]) else np.nanmin(run)]
            for tmp in run[1:]:
                tmp = tmp if not np.isnan(tmp) else np.nanmin(run[1:])
                val.append(0.9 * val[-1] + 0.1 * tmp)
            smoothed.append(val)
        mean_return = np.mean(np.asarray(smoothed), axis=0)
        std_return = np.std(np.asarray(smoothed), axis=0)
        lower_bound = mean_return - 1.96 * std_return / math.sqrt(100)
        upper_bound = mean_return + 1.96 * std_return / math.sqrt(100)
        plt.fill_between(np.arange(len(mean_return)),
                         lower_bound,
                         upper_bound,
                         alpha=0.25
                         )
        plt.axhline(0.961, linestyle='--', label="optimal", c="magenta")
        plt.plot(np.arange(len(mean_return)),
                 mean_return,
                 alpha=1,
                 label="OFU",
                 linewidth=3
                 )
    # plt.fill_between(np.arange(len(mean_return)),
    #                  19.03 - 4.62,
    #                  19.03 + 4.62,
    #                  alpha=0.25,
    #                  color="magenta"
    #                  )
    plt.xlabel("every 100 training steps")
    plt.ylabel("discounted (test?) return")
    plt.title(f" performance over {100} runs")
    plt.grid()

    # for k in range(10, 20):
    plt.plot(np.arange(len(mean_return)),
             smoothed[57],
             alpha=1,
             label="seed " + str(158),
             linewidth=3
             )
    plt.legend()
    plt.show()
