import matplotlib.pyplot as plt
import math
import numpy as np
import itertools
from matplotlib import ticker

plt.style.use('ggplot')

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
monitor = "NExpert",  # "Full", "RandomNonZero", "Ask", "Button", "N", "Level"  # , "Random"
env = (
    "RiverSwim-6-v0",
    # "Gridworld-Penalty-3x3-v0",
    # "Gridworld-OneWay",
    # "Gridworld-Empty",
    # "Gridworld-TwoRoom-3x5",
    # "Gridworld-Hazard",
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
                             "Button": (0.672, "optimal"),
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
                                               "Button": (0.849, "optimal"),
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
                               "N": (0.826, "optimal"),
                               "Random": (0.826, "optimal"),
                               "RandomNonZero": (0.826, "optimal"),
                               "Full": (0.826, "optimal"),
                               },

        "Gridworld-TwoRoom_3x5": {"Ask": (0.941, "optimal"),
                                  "Button": (0.826, "optimal"),
                                  "Level": (0.941, "optimal"),
                                  "N": (0.941, "optimal"),
                                  "Random": (0.941, "optimal"),
                                  "RandomNonZero": (0.941, "optimal"),
                                  "Full": (0.941, "optimal"),
                                  },
        "Gridworld-TwoRoom_2x11": {"Ask": (0.941, "optimal"),
                                   "Button": (0.8261, "optimal"),
                                   "Level": (0.941, "optimal"),
                                   "N": (0.941, "optimal"),
                                   "Random": (0.941, "optimal"),
                                   "RandomNonZero": (0.941, "optimal"),
                                   "Full": (0.941, "optimal"),
                                   },
        }

for env, monitor in env_mon_combo:
    fig, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    algos = [
        (f"{monitor}", "blue", ""),
        # (f"{monitor}_1", "blue", "100%"),
        # (f"{monitor}_0.75", "red", "75%"),
        # (f"{monitor}_0.5", "green", "50%"),
        # (f"{monitor}_0.25", "orange", "25%"),
        # (f"{monitor}_0.1", "brown", "10%"),
        # (f"{monitor}_0.01", "magenta", "1%")
    ]

    assert n_runs == 30

    for conf in algos:
        algo, color, legend = conf
        ref, opt_caut = info[env][monitor]
        my_runs = []
        s_runs = []
        som_runs = []
        svm_runs = []
        for i in range(n_runs):
            x = np.load(f"data/neurips/mine/Gym-Grid/{env}/{algo}/data_{i}.npz")["test_return"]
            my_runs.append(x)
            x = np.load(f"data/neurips/Simone/iGym-Grid/{env}/{algo}/q_visit_-10.0_-10.0_1.0_1.0_1.0_0.0_0.01_{i}.npz")[
                "test/return"]
            s_runs.append(x)
        # print(np.argmin(np.array(my_runs).sum(-1)))
        # exit()
        my_smoothed = []
        s_smoothed = []

        for run in my_runs:
            val = [run[0]]
            for tmp in run[1:]:
                val.append(0.9 * val[-1] + 0.1 * tmp)
            my_smoothed.append(val)

        for run in s_runs:
            val = [run[0]]
            for tmp in run[1:]:
                val.append(0.9 * val[-1] + 0.1 * tmp)
            s_smoothed.append(val)

        my_mean_return = np.mean(np.asarray(my_smoothed), axis=0)
        my_std_return = np.std(np.asarray(my_smoothed), axis=0)
        my_lower_bound = my_mean_return - 1.96 * my_std_return / math.sqrt(n_runs)
        my_upper_bound = my_mean_return + 1.96 * my_std_return / math.sqrt(n_runs)
        ax.fill_between(np.arange(len(my_mean_return)),
                        my_lower_bound,
                        my_upper_bound,
                        alpha=0.25,
                        color=color
                        )
        ax.plot(np.arange(len(my_mean_return)),
                my_mean_return,
                alpha=1,
                linewidth=4,
                c=color,
                )

        s_mean_return = np.mean(np.asarray(s_smoothed), axis=0)
        s_std_return = np.std(np.asarray(s_smoothed), axis=0)
        s_lower_bound = s_mean_return - 1.96 * s_std_return / math.sqrt(n_runs)
        s_upper_bound = s_mean_return + 1.96 * s_std_return / math.sqrt(n_runs)
        ax.fill_between(np.arange(len(s_mean_return)),
                        s_lower_bound,
                        s_upper_bound,
                        alpha=0.25,
                        color="red"
                        )
        ax.plot(np.arange(len(s_mean_return)),
                s_mean_return,
                alpha=1,
                linewidth=4,
                c="red",
                )

        plt.axhline(ref, linestyle="--", color="k", linewidth=3, label=f"{opt_caut}")
        # ax.set_ylabel("Discounted Test Return", weight="bold", fontsize=18)
        # ax.legend(loc='lower right', ncol=2, bbox_to_anchor=(1, 0))
        ax.xaxis.set_tick_params(labelsize=20, colors="black")
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 10:.0f}"))
        # plt.title(f"{env}_{monitor}")
        # plt.xlabel("Steps (x$10^3$)", weight="bold", fontsize=30)
        ax.xaxis.label.set_color('black')
        ax.set_xticks(np.arange(0, 201, 40))
        # ax.set_xticklabels([])
        ax.set_xlim(0, 210)
        ax.yaxis.set_tick_params(labelsize=20, colors="black")
        # ax.yaxis.label.set_color('black')
        # ax.set_ylim(0, 20)

        # if monitor == "Full":
        # ax.set_ylabel("Discounted test return",
        #               weight="bold",
        #               fontsize=20,
        #               # rotation="horizontal",
        #               # labelpad=50,
        #               # ha='right'
        #               )
        # ax.legend(loc='lower right', bbox_to_anchor=(1, 0))
        # ax.set_yticks([0.2, 0.5, 0.8, 1])
        # else:
        # ax.set_yticklabels([])
        # elif monitor == "Button":
        #     ax.set_ylabel("Discounted test return",
        #                   weight="bold",
        #                   fontsize=20,
        #                   # rotation="horizontal",
        #                   # labelpad=50,
        #                   # ha='center'
        #                   )

    # plt.show()
    plt.savefig(f"/Users/alirezakazemipour/Desktop/{monitor}_{env}.pdf",
                format="pdf",
                bbox_inches="tight"
                )
    plt.close()
