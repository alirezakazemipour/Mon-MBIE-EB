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
# plt.rc('figure', titlesize=BIGGER_SIZE)
# plt.title(f"EOP", weight="bold")

n_runs = 30
monitor = "Random", #"Full", "Button", "Ask", "NSupporter", "Level",  # "Random"
env = (
    # "RiverSwim-6-v0",
    # "Gridworld-Penalty-3x3-v0",
    # "Gridworld-Corridor-3x4-v0",
    # "Gridworld-Empty-Distract-6x6-v0",
    # "Gridworld-Ultimate-Snake-4x4-v0",
    "Gridworld-Snake-6x6-v0",
    # "Gridworld-Bypass-3x5-v0",

)
env_mon_combo = itertools.product(env, monitor)

info = {"RiverSwim-6-v0": {"Ask": (20.02, "Optimal"),
                           "Button": (19.14, "Optimal"),
                           "Level": (19.83, "Optimal"),
                           "N": (20.02, "Optimal"),
                           "Random": (20.02, "Optimal"),
                           "RandomNonZero": (20.02, "Optimal"),
                           "Full": (20.02, "Optimal"),
                           },
        "Gridworld-Empty-Distract-6x6-v0": {"Ask": (0.904, "Optimal"),
                                            "Button": (0.19, "Optimal"),
                                            "Level": (0.904, "Optimal"),
                                            "NSupporter": (0.915, "Optimal"),
                                            "NExpert": (0.904, "Optimal"),
                                            "Random": (0.904, "Optimal"),
                                            "RandomNonZero": (0.904, "Optimal"),
                                            "Full": (0.904, "Optimal"),
                                            },
        "Gridworld-Corridor-3x4-v0": {"Ask": (0.764, "Optimal"),
                                      "Button": (0.672, "Optimal"),
                                      "Level": (0.764, "Optimal"),
                                      "N": (0.764, "Optimal"),
                                      "Random": (0.764, "Optimal"),
                                      "RandomNonZero": (0.764, "Optimal"),
                                      "Full": (0.764, "Optimal"),
                                      },
        "Gridworld-Penalty-3x3-v0": {"Ask": (0.941, "Optimal"),
                                     "Button": (0.849, "Optimal"),
                                     "Level": (0.941, "Optimal"),
                                     "N": (0.941, "Optimal"),
                                     "Random": (0.941, "Optimal"),
                                     "RandomNonZero": (0.941, "Optimal"),
                                     "Full": (0.941, "Optimal"),
                                     },
        "Gridworld-Bypass-3x5-v0": {"Ask": (0.904, "Cautious"),
                                    "Button": (0.308, "Cautious"),
                                    "Level": (0.904, "Cautious"),
                                    "NSupporter": (0.91, "Cautious"),
                                    "NExpert": (0.904, "Cautious"),
                                    "Random": (0.904, "Cautious"),
                                    "RandomNonZero": (0.904, "Cautious"),
                                    "Full": (0.941, "Optimal"),
                                    },
        "Gridworld-Snake-6x6-v0": {"Ask": (0.904, "Cautious"),
                                   "Button": (0.19, "Cautious"),
                                   "Level": (0.904, "Cautious"),
                                   "NSupporter": (0.915, "Cautious"),
                                   "NExpert": (0.904, "Cautious"),
                                   "Random": (0.904, "Cautious"),
                                   "RandomNonZero": (0.904, "Cautious"),
                                   "Full": (0.904, "Cautious"),
                                   },
        }

for env, monitor in env_mon_combo:
    fig, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    algos = [
        (f"{monitor}", "blue", "0.5"),
        # (f"{monitor}_0.75", "red", "75%"),
        # (f"{monitor}_0.5", "green", "50%"),
        # (f"{monitor}_0.25", "orange", "25%"),
        # (f"{monitor}_0.1", "brown", "10%"),
        # (f"{monitor}_0.01", "magenta", "1%")
    ]

    assert n_runs == 30

    for conf in algos:
        algo, color, prob = conf
        ref, opt_caut = info[env][monitor]
        my_runs = []
        s_runs = []
        som_runs = []
        svm_runs = []
        for i in range(n_runs):
            x = np.load(f"data/stochastically_observable/mine/"
                        f"Gym-Grid/{env}/{algo}_{prob}/data_{i}.npz")["test_return"]
            my_runs.append(x)

            x = np.load(
                f"data/stochastically_observable/simone/"
                f"iGym-Grid/{env}/{algo}Monitor_{prob}/{monitor}Monitor__{prob}_{i}.npz")[
                "test/return"]
            s_runs.append(x)

        # print(np.argmin(np.array(my_runs).sum(-1)))
        # exit()
        my_smoothed = []
        s_smoothed = []
        som_smoothed = []
        svm_smoothed = []

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
                label="Double MBIE"
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
                label="Directed-E$\mathbf{^2}$"
                )

        plt.axhline(ref, linestyle="--", color="k", linewidth=3, label=f"{opt_caut}")
        # ax.set_ylabel("Discounted Test Return", weight="bold", fontsize=18)
        # ax.legend(loc='lower right', ncol=2, bbox_to_anchor=(1, 0))
        ax.xaxis.set_tick_params(labelsize=20, colors="black")
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 10:.0f}"))
        # plt.xlabel("Training Steps (x$10^3$)", weight="bold", fontsize=20)
        ax.xaxis.label.set_color('black')
        # ax.set_xticks(np.arange(0, 201, 100))
        ax.set_xticklabels([])
        ax.set_xlim(0, 200)
        # ax.set_yticks(np.arange(np.min(my_mean_return) - 0.05 * (np.max(my_mean_return) - np.min(my_mean_return)),
        #                         ref + 0.1 * ref,
        #                         (np.max(my_mean_return) - np.min(my_mean_return)) / 5
        #                         )
        #               )
        # ax.set_ylim([np.min(my_mean_return) - 0.05 * (np.max(my_mean_return) - np.min(my_mean_return)),
        #                         ref + 0.05 * (np.max(my_mean_return) - np.min(my_mean_return))])
        ax.yaxis.set_tick_params(labelsize=20, colors="black")
        # ax.yaxis.label.set_color('black')
        ax.set_ylim(0, 1)

        if monitor == "Random":
            # ax.set_ylabel("Discounted Test Return",
            #               weight="bold",
            #               fontsize=20,
            #               # rotation="horizontal",
            #               # labelpad=50,
            #               # ha='right'
            #               color="k"
            #               )
            # ax.legend(loc='lower right', bbox_to_anchor=(1, 0))
            ax.set_yticks([0, 0.2, 0.5, 0.8, 1])
        else:
            ax.set_yticklabels([])
        # if monitor == "Button":
        #     ax.set_ylabel("Discounted test return",
        #                   weight="bold",
        #                   fontsize=20,
        #                   # rotation="horizontal",
        #                   # labelpad=50,
        #                   # ha='center'
        #                   )
        #     ax.set_ylim(-0.8, 0.25)

    # plt.tight_layout()
    # plt.show()
    plt.savefig(f"/Users/alirezakazemipour/Desktop/{monitor}_{env}_{prob}.pdf",
                format="pdf",
                bbox_inches="tight"
                )
    plt.close()
