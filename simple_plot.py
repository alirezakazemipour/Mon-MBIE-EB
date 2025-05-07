import matplotlib.pyplot as plt
import math
import numpy as np
import itertools
from matplotlib import ticker
import os

mon_mbie_eb_color = "#2a9d8f"
de2_color = "#f4a261"

n_runs = 30
monitor = "Full", "RandomNonZero", "Ask", "Button", "N", "Level"  # , "Random"
env = (
    "RiverSwim-6-v0",
    "Gridworld-OneWay",
    "Gridworld-TwoRoom-3x5",
    "Gridworld-TwoRoom-2x11",
    "Gridworld-Hazard",
    "Gridworld-Corridor",
    "Gridworld-Loop",
    "Gridworld-Empty",
)
env_mon_combo = itertools.product(env, monitor)

info = {"RiverSwim-6-v0": {"Ask": (20.02, "Minimax-Optimal"),
                           "Button": (19.14, "Minimax-Optimal"),
                           "Level": (19.83, "Minimax-Optimal"),
                           "N": (20.02, "Minimax-Optimal"),
                           "NExpert": (20.02, "Minimax-Optimal"),
                           "Random": (20.02, "Minimax-Optimal"),
                           "RandomNonZero": (20.02, "Minimax-Optimal"),
                           "Full": (20.02, "Minimax-Optimal"),
                           },
        "Gridworld-Empty": {"Ask": (0.904, "Minimax-Optimal"),
                            "Button": (0.799, "Minimax-Optimal"),
                            "Level": (0.904, "Minimax-Optimal"),
                            "N": (0.915, "Minimax-Optimal"),
                            "Random": (0.904, "Minimax-Optimal"),
                            "RandomNonZero": (0.904, "Minimax-Optimal"),
                            "Full": (0.904, "Minimax-Optimal"),
                            },
        "Gridworld-OneWay": {"Ask": (0.764, "Minimax-Optimal"),
                             "Button": (0.66, "Minimax-Optimal"),
                             "Level": (0.764, "Minimax-Optimal"),
                             "N": (0.764, "Minimax-Optimal"),
                             "Random": (0.764, "Minimax-Optimal"),
                             "RandomNonZero": (0.764, "Minimax-Optimal"),
                             "Full": (0.764, "Minimax-Optimal"),
                             },
        "Gridworld-Penalty-3x3-v0": {"Ask": (0.941, "Minimax-Optimal"),
                                     "Button": (0.849, "Minimax-Optimal"),
                                     "Level": (0.941, "Minimax-Optimal"),
                                     "N": (0.941, "Minimax-Optimal"),
                                     "Random": (0.941, "Minimax-Optimal"),
                                     "RandomNonZero": (0.941, "Minimax-Optimal"),
                                     "Full": (0.941, "Minimax-Optimal"),
                                     },
        "Gridworld-TwoRoom-Quicksand-3x5-v0": {"Ask": (0.941, "Minimax-Optimal"),
                                               "Button": (0.9, "Minimax-Optimal"),
                                               "Level": (0.941, "Minimax-Optimal"),
                                               "N": (0.941, "Minimax-Optimal"),
                                               "Random": (0.941, "Minimax-Optimal"),
                                               "RandomNonZero": (0.941, "Minimax-Optimal"),
                                               "Full": (0.941, "Minimax-Optimal"),
                                               },
        "Gridworld-Hazard": {"Ask": (0.914, "Minimax-Optimal"),
                             "Button": (0.821, "Minimax-Optimal"),
                             "Level": (0.914, "Minimax-Optimal"),
                             "N": (0.914, "Minimax-Optimal"),
                             "Random": (0.914, "Minimax-Optimal"),
                             "RandomNonZero": (0.914, "Minimax-Optimal"),
                             "Full": (0.914, "Minimax-Optimal"),
                             },
        "Gridworld-Loop": {"Ask": (0.961, "Minimax-Optimal"),
                           "Button": (0.845, "Minimax-Optimal"),
                           "Level": (0.961, "Minimax-Optimal"),
                           "N": (0.961, "Minimax-Optimal"),
                           "Random": (0.961, "Minimax-Optimal"),
                           "RandomNonZero": (0.961, "Minimax-Optimal"),
                           "Full": (0.961, "Minimax-Optimal"),
                           },

        "Gridworld-Corridor": {"Ask": (0.826, "Minimax-Optimal"),
                               "Button": (0.712, "Minimax-Optimal"),
                               "Level": (0.826, "Minimax-Optimal"),
                               "N": (0.846, "Minimax-Optimal"),
                               "Random": (0.826, "Minimax-Optimal"),
                               "RandomNonZero": (0.826, "Minimax-Optimal"),
                               "Full": (0.826, "Minimax-Optimal"),
                               },

        "Gridworld-TwoRoom-3x5": {"Ask": (0.941, "Minimax-Optimal"),
                                  "Button": (0.838, "Minimax-Optimal"),
                                  "Level": (0.941, "Minimax-Optimal"),
                                  "N": (0.941, "Minimax-Optimal"),
                                  "Random": (0.941, "Minimax-Optimal"),
                                  "RandomNonZero": (0.941, "Minimax-Optimal"),
                                  "Full": (0.941, "Minimax-Optimal"),
                                  },
        "Gridworld-TwoRoom-2x11": {"Ask": (0.941, "Minimax-Optimal"),
                                   "Button": (0.8261, "Minimax-Optimal"),
                                   "Level": (0.941, "Minimax-Optimal"),
                                   "N": (0.941, "Minimax-Optimal"),
                                   "Random": (0.941, "Minimax-Optimal"),
                                   "RandomNonZero": (0.941, "Minimax-Optimal"),
                                   "Full": (0.941, "Minimax-Optimal"),
                                   },
        }

assert n_runs == 30
for env, monitor in env_mon_combo:
    fig, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ref, ref_label = info[env][monitor]
    mon_mbie_eb_runs = []
    dee_runs = []
    for i in range(n_runs):
        x = np.load(f"data/Mon_MBIE_EB/{env}/{monitor}/data_{i}.npz")["test_return"]
        mon_mbie_eb_runs.append(x)
        x = np.load(f"data/DEE/{env}/{monitor}/data_{i}.npz")["test/return"]
        dee_runs.append(x)
    mon_mbie_eb_smoothed = []
    dee_smoothed = []

    for run in mon_mbie_eb_runs:
        val = [run[0]]
        for tmp in run[1:]:
            val.append(0.9 * val[-1] + 0.1 * tmp)
        mon_mbie_eb_smoothed.append(val)

    for run in dee_runs:
        val = [run[0]]
        for tmp in run[1:]:
            val.append(0.9 * val[-1] + 0.1 * tmp)
        dee_smoothed.append(val)

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

    dee_mean_return = np.mean(np.asarray(dee_smoothed), axis=0)
    dee_std_return = np.std(np.asarray(dee_smoothed), axis=0)
    dee_lower_bound = dee_mean_return - 1.96 * dee_std_return / math.sqrt(n_runs)
    dee_upper_bound = dee_mean_return + 1.96 * dee_std_return / math.sqrt(n_runs)
    ax.fill_between(np.arange(len(dee_mean_return)),
                    dee_lower_bound,
                    dee_upper_bound,
                    alpha=0.25,
                    color=de2_color
                    )
    ax.plot(np.arange(len(dee_mean_return)),
            dee_mean_return,
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

    ax.set_xticks(np.arange(0, 201, 40))

    if env != "RiverSwim-6-v0":
        ax.set_yticks([0, 0.2, 0.5, 0.8, 1])
        ax.set_ylim([0 ,1])
    else:
        ax.set_yticks([0, 5, 10, 15, 20])
        ax.set_ylim([0, 22])

    os.makedirs("figs", exist_ok=True)
    plt.savefig(f"figs/{monitor}_{env}.pdf",
                format="pdf",
                bbox_inches="tight",
                dpi=300
                )
    plt.close()
