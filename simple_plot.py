import matplotlib.pyplot as plt
import math
import numpy as np
import itertools
from matplotlib import ticker
import os
import warnings

de2_exists = True
if not os.path.exists("data/DEE"):
    de2_exists = False
    warnings.warn("Directed-E$^2$ data not found. Ignoring it.")

mon_mbie_eb_color = "#2a9d8f"
de2_color = "#f4a261"

n_runs = 30
monitor = "MDP", "SemiRandom", "Ask", "Button", "NSupporters", "Level"  # , "Random"
env = (
    "RiverSwim",
    "Gridworld-OneWay",
    "Gridworld-TwoRoom-3x5",
    "Gridworld-TwoRoom-2x11",
    "Gridworld-Hazard",
    "Gridworld-Corridor",
    "Gridworld-Loop",
    "Gridworld-Empty",
)
env_mon_combo = itertools.product(env, monitor)

info = {"RiverSwim": {"Ask": (20.02, "Minimax-Optimal"),
                      "Button": (19.14, "Minimax-Optimal"),
                      "Level": (19.83, "Minimax-Optimal"),
                      "NSupporters": (20.02, "Minimax-Optimal"),
                      "NExpert": (20.02, "Minimax-Optimal"),
                      "Random": (20.02, "Minimax-Optimal"),
                      "SemiRandom": (20.02, "Minimax-Optimal"),
                      "MDP": (20.02, "Minimax-Optimal"),
                      },
        "Gridworld-Empty": {"Ask": (0.904, "Minimax-Optimal"),
                            "Button": (0.799, "Minimax-Optimal"),
                            "Level": (0.904, "Minimax-Optimal"),
                            "NSupporters": (0.915, "Minimax-Optimal"),
                            "Random": (0.904, "Minimax-Optimal"),
                            "SemiRandom": (0.904, "Minimax-Optimal"),
                            "MDP": (0.904, "Minimax-Optimal"),
                            },
        "Gridworld-OneWay": {"Ask": (0.764, "Minimax-Optimal"),
                             "Button": (0.66, "Minimax-Optimal"),
                             "Level": (0.764, "Minimax-Optimal"),
                             "NSupporters": (0.764, "Minimax-Optimal"),
                             "Random": (0.764, "Minimax-Optimal"),
                             "SemiRandom": (0.764, "Minimax-Optimal"),
                             "MDP": (0.764, "Minimax-Optimal"),
                             },
        "Gridworld-Hazard": {"Ask": (0.914, "Minimax-Optimal"),
                             "Button": (0.821, "Minimax-Optimal"),
                             "Level": (0.914, "Minimax-Optimal"),
                             "NSupporters": (0.914, "Minimax-Optimal"),
                             "Random": (0.914, "Minimax-Optimal"),
                             "SemiRandom": (0.914, "Minimax-Optimal"),
                             "MDP": (0.914, "Minimax-Optimal"),
                             },
        "Gridworld-Loop": {"Ask": (0.961, "Minimax-Optimal"),
                           "Button": (0.845, "Minimax-Optimal"),
                           "Level": (0.961, "Minimax-Optimal"),
                           "NSupporters": (0.961, "Minimax-Optimal"),
                           "Random": (0.961, "Minimax-Optimal"),
                           "SemiRandom": (0.961, "Minimax-Optimal"),
                           "MDP": (0.961, "Minimax-Optimal"),
                           },

        "Gridworld-Corridor": {"Ask": (0.826, "Minimax-Optimal"),
                               "Button": (0.712, "Minimax-Optimal"),
                               "Level": (0.826, "Minimax-Optimal"),
                               "NSupporters": (0.846, "Minimax-Optimal"),
                               "Random": (0.826, "Minimax-Optimal"),
                               "SemiRandom": (0.826, "Minimax-Optimal"),
                               "MDP": (0.826, "Minimax-Optimal"),
                               },

        "Gridworld-TwoRoom-3x5": {"Ask": (0.922, "Minimax-Optimal"),
                                  "Button": (0.835, "Minimax-Optimal"),
                                  "Level": (0.922, "Minimax-Optimal"),
                                  "NSupporters": (0.922, "Minimax-Optimal"),
                                  "Random": (0.922, "Minimax-Optimal"),
                                  "SemiRandom": (0.922, "Minimax-Optimal"),
                                  "MDP": (0.922, "Minimax-Optimal"),
                                  },
        "Gridworld-TwoRoom-2x11": {"Ask": (0.941, "Minimax-Optimal"),
                                   "Button": (0.8261, "Minimax-Optimal"),
                                   "Level": (0.941, "Minimax-Optimal"),
                                   "NSupporters": (0.941, "Minimax-Optimal"),
                                   "Random": (0.941, "Minimax-Optimal"),
                                   "SemiRandom": (0.941, "Minimax-Optimal"),
                                   "MDP": (0.941, "Minimax-Optimal"),
                                   },
        }

assert n_runs == 30
for env, monitor in env_mon_combo:
    fig, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ref, ref_label = info[env][monitor]
    mon_mbie_eb_runs = []
    de2_runs = []
    for i in range(n_runs):
        x = np.load(f"data/Mon_MBIE_EB/{env}/{monitor}/data_{i}.npz")["test_return"]
        mon_mbie_eb_runs.append(x)
        if de2_exists:
            x = np.load(f"data/de2/{env}/{monitor}/data_{i}.npz")["test/return"]
            de2_runs.append(x)
    mon_mbie_eb_smoothed = []
    de2_smoothed = []

    for run in mon_mbie_eb_runs:
        val = [run[0]]
        for tmp in run[1:]:
            val.append(0.9 * val[-1] + 0.1 * tmp)
        mon_mbie_eb_smoothed.append(val)

    if de2_exists:
        for run in de2_runs:
            val = [run[0]]
            for tmp in run[1:]:
                val.append(0.9 * val[-1] + 0.1 * tmp)
            de2_smoothed.append(val)

    mon_mbie_eb_mean_return = np.mean(np.asarray(mon_mbie_eb_smoothed), axis=0)
    mon_mbie_eb_std_return = np.std(np.asarray(mon_mbie_eb_smoothed), axis=0)
    mon_mbie_eb_lower_bound = mon_mbie_eb_mean_return - 1.96 * mon_mbie_eb_std_return / math.sqrt(n_runs)
    mon_mbie_eb_upper_bound = mon_mbie_eb_mean_return + 1.96 * mon_mbie_eb_std_return / math.sqrt(n_runs)
    ax.fill_between(np.arange(len(mon_mbie_eb_mean_return)),
                    mon_mbie_eb_lower_bound,
                    mon_mbie_eb_upper_bound,
                    alpha=0.25,
                    color=mon_mbie_eb_color
                    )
    ax.plot(np.arange(len(mon_mbie_eb_mean_return)),
            mon_mbie_eb_mean_return,
            alpha=1,
            linewidth=4,
            c=mon_mbie_eb_color,
            label="Mon-MBIE-EB"
            )

    if de2_exists:
        de2_mean_return = np.mean(np.asarray(de2_smoothed), axis=0)
        de2_std_return = np.std(np.asarray(de2_smoothed), axis=0)
        de2_lower_bound = de2_mean_return - 1.96 * de2_std_return / math.sqrt(n_runs)
        de2_upper_bound = de2_mean_return + 1.96 * de2_std_return / math.sqrt(n_runs)
        ax.fill_between(np.arange(len(de2_mean_return)),
                        de2_lower_bound,
                        de2_upper_bound,
                        alpha=0.25,
                        color=de2_color
                        )
        ax.plot(np.arange(len(de2_mean_return)),
                de2_mean_return,
                alpha=1,
                linewidth=4,
                c=de2_color,
                label="Directed-E$^2$"
                )

    plt.axhline(ref, linestyle="--", color="k", linewidth=3, label=f"{ref_label}")
    ax.set_ylabel("Discounted Test Return")
    ax.legend(loc='lower right', ncol=2, bbox_to_anchor=(1, 0))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 10:.0f}"))
    plt.title(f"{env}_{monitor}")
    plt.xlabel("Training Steps (x$10^3$)")

    ax.set_xlim([0, 200])
    ax.set_xticks(np.arange(0, 201, 40))

    if env != "RiverSwim":
        ax.set_yticks([0, 0.2, 0.5, 0.8, 1])
        ax.set_ylim([0, 1])
    else:
        ax.set_yticks([0, 5, 10, 15, 20])
        ax.set_ylim([0, 22])

    os.makedirs("figs", exist_ok=True)
    plt.savefig(f"figs/{env}_{monitor}.pdf",
                format="pdf",
                bbox_inches="tight",
                dpi=300
                )
    plt.close()
