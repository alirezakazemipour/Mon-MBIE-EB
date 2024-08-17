import gymnasium
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import numpy as np
from tqdm import tqdm
from src.actor import Greedy
from src.critic import MonQCritic
from src.experiment import MonExperiment
from src.wrappers import monitor_wrappers
import pickle


@hydra.main(version_base=None, config_path="configs", config_name="default")
def run(cfg: DictConfig) -> None:
    env = gymnasium.make(**cfg.environment)
    env_test = gymnasium.make(**cfg.environment)
    env = getattr(monitor_wrappers, cfg.monitor.id)(env,
                                                    **{**cfg.monitor, **cfg.environment.get("monitor", {})}
                                                    )
    env_test = getattr(monitor_wrappers, cfg.monitor.id)(env_test,
                                                         **{**cfg.monitor, **cfg.environment.monitor},
                                                         test=True
                                                         )

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
    #         a = {"env": 1, "mon": 0}
    #         obs, r, term, trunc, _ = env.step(a)
    #         ret_e += (0.99 ** t)*(r["env"] + r["mon"])
    #         if term or trunc:
    #             ret.append(ret_e)
    #             break
    #         t += 1
    #
    # print(np.mean(ret))
    # print(np.std(ret))
    # exit()

    critic = MonQCritic(env.observation_space["env"].n,
                        env.observation_space["mon"].n,
                        env.action_space["env"].n,
                        env.action_space["mon"].n,
                        **cfg.environment.critic,
                        )
    actor = Greedy(critic)
    experiment = MonExperiment(env,
                               env_test,
                               actor,
                               critic,
                               **{**cfg.environment.experiment, **cfg.experiment}
                               )
    data = experiment.train()
    print(f"total episodes: {experiment.tot_episodes}")
    print("visits:", critic.env_visit.astype(int))
    print("observs:", critic.env_obsrv_count.astype(int))  # noqa
    print("rwd model:", critic.env_rwd_model)

    if cfg.experiment.datadir is not None:
        filepath = os.path.join(cfg.experiment.datadir,
                                cfg.environment.id,
                                cfg.monitor.id + "_" + str(cfg.monitor.prob)
                                )
        os.makedirs(filepath, exist_ok=True)
        seed = str(cfg.experiment.rng_seed)
        savepath = os.path.join(filepath, f"data_{seed}.npz")
        np.savez(savepath, **data)

        if not os.path.isfile(savepath):
            with open(savepath, 'wb') as f:
                pickle.dump(cfg, f)


if __name__ == "__main__":
    run()
    exit()
