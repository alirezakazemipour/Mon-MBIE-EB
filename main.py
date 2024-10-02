import gymnasium
import hydra
from omegaconf import DictConfig
import os
import numpy as np
from src.actor import Greedy
from src.critic import MonQCritic
from src.experiment import MonExperiment
from src.wrappers import monitor_wrappers
from src.utils import report_river_swim
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

    # report_river_swim(env_test)
    # if "RiverSwim-6-v0" in cfg.environment["id"]:
    #     cfg.critic.vi_iter = 800

    critic = MonQCritic(env.observation_space["env"].n,
                        env.observation_space["mon"].n,
                        env.action_space["env"].n,
                        env.action_space["mon"].n,
                        **{**cfg.critic, **cfg.environment.critic},
                        )
    actor = Greedy(critic)
    experiment = MonExperiment(env,
                               env_test,
                               actor,
                               critic,
                               **{**cfg.environment.experiment, **cfg.experiment}
                               )
    data = experiment.train()
    print(f"\ntotal episodes: {experiment.tot_episodes}")
    print(f"\nexplore episodes: {experiment.explore_episodes}")
    print("\nvisits:", critic.env_visit.astype(int))
    print("\nobservs:", critic.env_obsrv_count.astype(int))  # noqa
    print("\nrwd model:", critic.env_rwd_model)
    # print("\njoint count: ", critic.joint_count[-1])
    print("\nmon rwd: ", critic.mon_rwd_model)
    # print("\ndynamics: ", critic.joint_dynamics)
    # print(critic.joint_q)

    if cfg.experiment.datadir is not None:
        filepath = os.path.join(cfg.experiment.datadir,
                                cfg.environment.id,
                                cfg.monitor.id + "_" + str(cfg.monitor.prob)
                                )
        os.makedirs(filepath, exist_ok=True)
        seed = str(cfg.experiment.rng_seed)
        savepath = os.path.join(filepath, f"data_{seed}")
        np.savez(savepath + ".npz", **data)

        if not os.path.isfile(savepath):
            with open(savepath + ".pkl", 'wb') as f:
                pickle.dump(cfg, f)


if __name__ == "__main__":
    run()
    exit()
