import matplotlib.pyplot as plt
import math
import numpy as np
import itertools
from matplotlib import ticker
import os

mon_mbie_eb_color = "#2a9d8f"
de2_color = "#f4a261"

SMALL_SIZE = 8
MEDIUM_SIZE = 24
BIGGER_SIZE = 26

plt.rc('font', size=SMALL_SIZE, weight='bold')  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=17)  # legend fontsize

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

info = {"RiverSwim-6-v0": {"Ask": (20.02, "optimal"),
                           "Button": (19.14, "optimal"),
                           "Level": (19.83, "optimal"),
                           "N": (20.02, "optimal"),
                           "NExpert": (20.02, "optimal"),
                           "Random": (20.02, "optimal"),
                           "RandomNonZero": (20.02, "optimal"),
                           "Full": (20.02, "optimal"),
                           },
        "Gridworld-Empty": {"Ask": (0.904, "optimal"),
                            "Button": (0.799, "optimal"),
                            "Level": (0.904, "optimal"),
                            "N": (0.915, "optimal"),
                            "Random": (0.904, "optimal"),
                            "RandomNonZero": (0.904, "optimal"),
                            "Full": (0.904, "optimal"),
                            },
        "Gridworld-OneWay": {"Ask": (0.764, "optimal"),
                             "Button": (0.66, "optimal"),
                             "Level": (0.764, "optimal"),
                             "N": (0.764, "optimal"),
                             "Random": (0.764, "optimal"),
                             "RandomNonZero": (0.764, "optimal"),
                             "Full": (0.764, "optimal"),
                             },
        "Gridworld-Penalty-3x3-v0": {"Ask": (0.941, "optimal"),
                                     "Button": (0.849, "optimal"),
                                     "Level": (0.941, "optimal"),
                                     "N": (0.941, "optimal"),
                                     "Random": (0.941, "optimal"),
                                     "RandomNonZero": (0.941, "optimal"),
                                     "Full": (0.941, "optimal"),
                                     },
        "Gridworld-TwoRoom-Quicksand-3x5-v0": {"Ask": (0.941, "optimal"),
                                               "Button": (0.9, "optimal"),
                                               "Level": (0.941, "optimal"),
                                               "N": (0.941, "optimal"),
                                               "Random": (0.941, "optimal"),
                                               "RandomNonZero": (0.941, "optimal"),
                                               "Full": (0.941, "optimal"),
                                               },
        "Gridworld-Hazard": {"Ask": (0.914, "optimal"),
                             "Button": (0.821, "optimal"),
                             "Level": (0.914, "optimal"),
                             "N": (0.914, "optimal"),
                             "Random": (0.914, "optimal"),
                             "RandomNonZero": (0.914, "optimal"),
                             "Full": (0.914, "optimal"),
                             },
        "Gridworld-Loop": {"Ask": (0.961, "optimal"),
                           "Button": (0.845, "optimal"),
                           "Level": (0.961, "optimal"),
                           "N": (0.961, "optimal"),
                           "Random": (0.961, "optimal"),
                           "RandomNonZero": (0.961, "optimal"),
                           "Full": (0.961, "optimal"),
                           },

        "Gridworld-Corridor": {"Ask": (0.826, "optimal"),
                               "Button": (0.712, "optimal"),
                               "Level": (0.826, "optimal"),
                               "N": (0.846, "optimal"),
                               "Random": (0.826, "optimal"),
                               "RandomNonZero": (0.826, "optimal"),
                               "Full": (0.826, "optimal"),
                               },

        "Gridworld-TwoRoom-3x5": {"Ask": (0.941, "optimal"),
                                  "Button": (0.838, "optimal"),
                                  "Level": (0.941, "optimal"),
                                  "N": (0.941, "optimal"),
                                  "Random": (0.941, "optimal"),
                                  "RandomNonZero": (0.941, "optimal"),
                                  "Full": (0.941, "optimal"),
                                  },
        "Gridworld-TwoRoom-2x11": {"Ask": (0.941, "optimal"),
                                   "Button": (0.8261, "optimal"),
                                   "Level": (0.941, "optimal"),
                                   "N": (0.941, "optimal"),
                                   "Random": (0.941, "optimal"),
                                   "RandomNonZero": (0.941, "optimal"),
                                   "Full": (0.941, "optimal"),
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
        x = np.load(f"data/DEE/{env}/{monitor}/q_visit_-10.0_-10.0_1.0_1.0_1.0_0.0_0.01_{i}.npz")[
                "test/return"]
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
            )

    plt.axhline(ref, linestyle="--", color="k", linewidth=3, label=f"{ref_label}")
    ax.set_ylabel("Discounted Test Return", weight="bold", fontsize=18)
    ax.legend(loc='lower right', ncol=2, bbox_to_anchor=(1, 0))
    ax.xaxis.set_tick_params(labelsize=20, colors="black")
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 10:.0f}"))
    plt.title(f"{env}_{monitor}")
    plt.xlabel("Steps (x$10^3$)", weight="bold", fontsize=30)
    ax.xaxis.label.set_color('black')
    ax.yaxis.set_tick_params(labelsize=20, colors="black")

    os.makedirs("figs", exist_ok=True)
    plt.savefig(f"figs/{monitor}_{env}.pdf",
                format="pdf",
                bbox_inches="tight",
                dpi=300
                )
    plt.close()
