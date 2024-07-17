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
    #         # while obs["mon"] == 1:
    #         #     a = {"env": 0, "mon": 0}
    #         #     obs, r, term, trunc, _ = env.step(a)
    #         #     ret_e += (0.99 ** t) * (r["env"] + r["mon"])
    #         #     t += 1
    #
    #         a = env.action_space.sample()
    #         a["env"] = 0
    #         obs, r, term, trunc, _ = env.step(a)
    #         ret_e += (0.99 ** t) * (r["env"] + r["mon"])
    #         if term or trunc:
    #             ret.append(ret_e)
    #             break
    #         t += 1

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
    # experiment.test()

    if cfg.experiment.datadir is not None:
        filepath = os.path.join(
            cfg.experiment.datadir,
            cfg.environment.id,
            cfg.monitor.id
        )
        os.makedirs(filepath, exist_ok=True)
        seed = str(cfg.experiment.rng_seed)
        savepath = os.path.join(filepath, "train_" + seed)
        np.save(savepath, np.array(return_train_history))
        savepath = os.path.join(filepath, "test_" + seed)
        np.save(savepath, np.array(return_test_history))

if __name__ == "__main__":
    # run()
    # exit()
    algos = ["FO", "NO"]
    for algo in algos:
        runs = []
        for i in range(30):
            x = np.load(f"data/{algo}/Gridworld-Empty-3x3-v0/Unsolvable/test_{i}.npy")
            runs.append(x)
        # print(np.argmin(np.nansum(np.asarray(runs), axis=-1)))
        # exit()
        smoothed = []
        for run in runs:
            val = [run[0]]
            for tmp in run[1:]:
                val.append(0.9 * val[-1] + 0.1 * tmp)
            smoothed.append(val)
        mean_return = np.mean(np.asarray(smoothed), axis=0)
        std_return = np.std(np.asarray(smoothed), axis=0)
        lower_bound = mean_return - 1.96 * std_return  / math.sqrt(len(runs))
        upper_bound = mean_return + 1.96 * std_return / math.sqrt(len(runs))
        plt.fill_between(np.arange(len(mean_return)),
                         lower_bound,
                         upper_bound,
                         alpha=0.25
                         )
        plt.plot(np.arange(len(mean_return)),
                 mean_return,
                 alpha=1,
                 label=algo,
                 linewidth=3
                 )
# plt.fill_between(np.arange(len(mean_return)),
#                  20 - 4.5,
#                  20 + 4.5,
#                  alpha=0.15,
#                  color="magenta"
#                  )
plt.axhline(.941, linestyle='--', label="optimal", c="magenta")
# plt.axhline(0.941, linestyle='--', label="cautious", c="olive")
plt.xlabel("training steps (x 100)")
plt.ylabel("discounted test return")
plt.title(f" performance over {10} runs")
plt.grid()
plt.legend()
plt.show()
