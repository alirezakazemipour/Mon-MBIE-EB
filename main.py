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
    # region Full
    if "MDP" in cfg.monitor.id:
        mon_rwd_model = np.zeros((env.observation_space["mon"].n,
                                  env.action_space["mon"].n
                                  )
                                 )

        monitor = np.ones((env.observation_space["env"].n *
                           env.observation_space["mon"].n *
                           env.action_space["env"].n,
                           env.action_space["mon"].n
                           )
                          )
        monitor.resize(env.observation_space["env"].n,
                       env.observation_space["mon"].n,
                       env.action_space["env"].n,
                       env.action_space["mon"].n
                       )

        mon_dynamics = np.ones((env.observation_space["env"].n,
                                env.observation_space["mon"].n,
                                env.action_space["env"].n,
                                env.action_space["mon"].n,
                                env.observation_space["mon"].n
                                )
                               )
    # endregion

    # region FullRandom
    if "FullRandom" == cfg.monitor.id:
        mon_rwd_model = np.zeros((env.observation_space["mon"].n, env.action_space["mon"].n))

        monitor = np.zeros((env.observation_space["env"].n,
                            env.observation_space["mon"].n,
                            env.action_space["env"].n,
                            env.action_space["mon"].n
                            )
                           ) + cfg.monitor.prob

        if cfg.environment.monitor.forbidden_states is not None:
            env_test.reset()
            for s in range(env.observation_space["env"].n):
                for a in range(env.action_space["env"].n):
                    state = {"env": s, "mon": 0}
                    env_test.set_state(state)
                    act = {"env": a, "mon": 0}
                    ns, *_ = env_test.step(act)
                    if ns["env"] in cfg.environment.monitor.forbidden_states:
                        monitor[s, :, a, :] = 0

        mon_dynamics = np.ones((env.observation_space["env"].n,
                                env.observation_space["mon"].n,
                                env.action_space["env"].n,
                                env.action_space["mon"].n,
                                env.observation_space["mon"].n
                                )
                               )
    # endregion

    # region SemiRandom
    if "SemiRandom" == cfg.monitor.id:
        mon_rwd_model = np.zeros((env.observation_space["mon"].n, env.action_space["mon"].n))

        monitor = np.zeros((env.observation_space["env"].n,
                            env.observation_space["mon"].n,
                            env.action_space["env"].n,
                            env.action_space["mon"].n
                            )
                           ) + cfg.monitor.prob

        if cfg.environment.monitor.forbidden_states is not None:
            env_test.reset()
            for s in range(env.observation_space["env"].n):
                for a in range(env.action_space["env"].n):
                    state = {"env": s, "mon": 0}
                    env_test.set_state(state)
                    act = {"env": a, "mon": 0}
                    ns, rwd, *_ = env_test.step(act)
                    if ns["env"] in cfg.environment.monitor.forbidden_states:
                        monitor[s, :, a, :] = 0

                    if rwd["env"] == 0 and ns["env"] not in cfg.environment.monitor.forbidden_states:
                        monitor[s, :, a, :] = 1

        mon_dynamics = np.ones((env.observation_space["env"].n,
                                env.observation_space["mon"].n,
                                env.action_space["env"].n,
                                env.action_space["mon"].n,
                                env.observation_space["mon"].n
                                )
                               )
    # endregion

    # region Ask
    if "Ask" in cfg.monitor.id:
        mon_rwd_model = np.zeros((env.observation_space["mon"].n, env.action_space["mon"].n))
        mon_rwd_model[:, 1] = -0.2

        monitor = np.zeros((env.observation_space["env"].n,
                            env.observation_space["mon"].n,
                            env.action_space["env"].n,
                            env.action_space["mon"].n
                            )
                           )
        monitor[..., 1] = cfg.monitor.prob
        if cfg.environment.monitor.forbidden_states is not None:
            env_test.reset()
            for s in range(env.observation_space["env"].n):
                for a in range(env.action_space["env"].n):
                    state = {"env": s, "mon": 0}
                    env_test.set_state(state)
                    act = {"env": a, "mon": 0}
                    ns, *_ = env_test.step(act)
                    if ns["env"] in cfg.environment.monitor.forbidden_states:
                        monitor[s, :, a, :] = 0

        mon_dynamics = np.ones((env.observation_space["env"].n,
                                env.observation_space["mon"].n,
                                env.action_space["env"].n,
                                env.action_space["mon"].n,
                                env.observation_space["mon"].n
                                )
                               )
    # endregion

    # region Button
    if "Button" in cfg.monitor.id:
        mon_rwd_model = np.zeros((env.observation_space["mon"].n, env.action_space["mon"].n))
        mon_rwd_model[1, :] = -0.2

        monitor = np.zeros((env.observation_space["env"].n,
                            env.observation_space["mon"].n,
                            env.action_space["env"].n,
                            env.action_space["mon"].n
                            )
                           )

        monitor[:, 1, ...] = cfg.monitor.prob

        if cfg.environment.monitor.forbidden_states is not None:
            env_test.reset()
            for s in range(env.observation_space["env"].n):
                for a in range(env.action_space["env"].n):
                    state = {"env": s, "mon": 1}
                    env_test.set_state(state)
                    act = {"env": a, "mon": 0}
                    ns, reward, terminated, *_ = env_test.step(act)
                    if ns["env"] in cfg.environment.monitor.forbidden_states:
                        monitor[s, :, a, :] = 0

        mon_dynamics = np.zeros((env.observation_space["env"].n,
                                 env.observation_space["mon"].n,
                                 env.action_space["env"].n,
                                 env.action_space["mon"].n,
                                 env.observation_space["mon"].n
                                 )
                                )
        mon_dynamics[:, 1, :, :, 1] = 1
        mon_dynamics[:, 0, :, :, 0] = 1

        button_cell = cfg.environment.monitor.button_cell_id
        button_flip_act = cfg.environment.monitor.button_flip_act

        mon_dynamics[button_cell, 1, button_flip_act, 0, 0] = 1
        mon_dynamics[button_cell, 1, button_flip_act, 0, 1] = 0
        mon_dynamics[button_cell, 0, button_flip_act, 0, 1] = 1
        mon_dynamics[button_cell, 0, button_flip_act, 0, 0] = 0
    # endregion

    # region NSupporters
    if "NSupporters" in cfg.monitor.id:
        mon_rwd_model = np.zeros((env.observation_space["mon"].n, env.action_space["mon"].n))
        mon_rwd_model = np.diag(np.diag(mon_rwd_model - 0.2 - 0.001)) + 0.001

        monitor = np.zeros((env.observation_space["env"].n,
                            env.observation_space["mon"].n,
                            env.action_space["env"].n,
                            env.action_space["mon"].n
                            )
                           )
        for i in range(env.observation_space["mon"].n):
            monitor[:, i, :, i] = cfg.monitor.prob

        if cfg.environment.monitor.forbidden_states is not None:
            env_test.reset()
            for s in range(env.observation_space["env"].n):
                for a in range(env.action_space["env"].n):
                    state = {"env": s, "mon": 0}
                    env_test.set_state(state)
                    act = {"env": a, "mon": 0}
                    ns, *_ = env_test.step(act)
                    if ns["env"] in cfg.environment.monitor.forbidden_states:
                        monitor[s, :, a, :] = 0

        mon_dynamics = np.ones((env.observation_space["env"].n,
                                env.observation_space["mon"].n,
                                env.action_space["env"].n,
                                env.action_space["mon"].n,
                                env.observation_space["mon"].n
                                )
                               ) / env.observation_space["mon"].n

    # endregion

    # region NExperts
    if "NExpert" in cfg.monitor.id:
        mon_rwd_model = np.zeros((env.observation_space["mon"].n, env.action_space["mon"].n))
        mon_rwd_model = np.diag(np.diag(mon_rwd_model - 0.2 + 0.001)) - 0.001
        mon_rwd_model = np.hstack([mon_rwd_model, np.zeros((env.observation_space["mon"].n, 1))])

        monitor = np.zeros((env.observation_space["env"].n,
                            env.observation_space["mon"].n,
                            env.action_space["env"].n,
                            env.action_space["mon"].n
                            )
                           )
        for i in range(env.observation_space["mon"].n):
            monitor[:, i, :, i] = cfg.monitor.prob

        if cfg.environment.monitor.forbidden_states is not None:
            env_test.reset()
            for s in range(env.observation_space["env"].n):
                for a in range(env.action_space["env"].n):
                    state = {"env": s, "mon": 0}
                    env_test.set_state(state)
                    act = {"env": a, "mon": 0}
                    ns, *_ = env_test.step(act)
                    if ns["env"] in cfg.environment.monitor.forbidden_states:
                        monitor[s, :, a, :] = 0

        mon_dynamics = np.ones((env.observation_space["env"].n,
                                env.observation_space["mon"].n,
                                env.action_space["env"].n,
                                env.action_space["mon"].n,
                                env.observation_space["mon"].n
                                )
                               ) / env.observation_space["mon"].n

    # endregion

    if "Level" in cfg.monitor.id:
        mon_rwd_model = np.zeros((env.observation_space["mon"].n, env.action_space["mon"].n)) - 0.2
        mon_rwd_model[..., -1] = 0

        monitor = np.zeros((env.observation_space["env"].n,
                            env.observation_space["mon"].n,
                            env.action_space["env"].n,
                            env.action_space["mon"].n
                            )
                           )

        monitor[:, -1, ...] = cfg.monitor.prob

        if cfg.environment.monitor.forbidden_states is not None:
            env_test.reset()
            for s in range(env.observation_space["env"].n):
                for a in range(env.action_space["env"].n):
                    state = {"env": s, "mon": 0}
                    env_test.set_state(state)
                    act = {"env": a, "mon": 0}
                    ns, reward, terminated, *_ = env_test.step(act)
                    if ns["env"] in cfg.environment.monitor.forbidden_states:
                        monitor[s, :, a, :] = 0

        mon_dynamics = np.zeros((env.observation_space["env"].n,
                                 env.observation_space["mon"].n,
                                 env.action_space["env"].n,
                                 env.action_space["mon"].n,
                                 env.observation_space["mon"].n
                                 )
                                )
        for level in range(cfg.monitor.n_levels - 1):
            mon_dynamics[:, level, :, level, level + 1] = 1

        mon_dynamics[:, -1, :, -2, -1] = 1

        for level in range(cfg.monitor.n_levels):
            mon_dynamics[:, level, :, -1, level] = 1

        for sm in range(env.observation_space["mon"].n):
            for am in range(env.action_space["mon"].n - 1):
                if sm != am:
                    mon_dynamics[:, sm, :, am, 0] = 1

    critic = MonQCritic(env.observation_space["env"].n,
                        env.observation_space["mon"].n,
                        env.action_space["env"].n,
                        env.action_space["mon"].n,
                        mon_rwd_model,
                        mon_dynamics,
                        monitor,
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
                                "Known_Monitor",
                                os.path.split(cfg.environment.id)[-1],
                                cfg.monitor.id + "_" + str(cfg.monitor.prob)
                                )
        os.makedirs(filepath, exist_ok=True)
        seed = str(cfg.experiment.rng_seed)
        savepath = os.path.join(filepath, f"data_{seed}")
        np.savez(savepath + ".npz", **data)


if __name__ == "__main__":
    run()
    exit()
