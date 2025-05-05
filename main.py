import gymnasium
import hydra
from omegaconf import DictConfig
import os
import numpy as np
from src.actor import Greedy
from src.critic import MonQCritic
from src.experiment import MonExperiment
from src.wrappers import monitor_wrappers


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

    if cfg.experiment.datadir is not None:
        filepath = os.path.join(cfg.experiment.datadir,
                                "Mon_MBIE_EB",
                                os.path.split(cfg.environment.id)[-1],
                                cfg.monitor.id
                                )
        os.makedirs(filepath, exist_ok=True)
        seed = str(cfg.experiment.rng_seed)
        savepath = os.path.join(filepath, f"data_{seed}")
        np.savez(savepath + ".npz", **data)


if __name__ == "__main__":
    run()
    exit()
