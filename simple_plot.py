import matplotlib.pyplot as plt
import math
import numpy as np
import itertools
from matplotlib import ticker

SMALL_SIZE = 8
MEDIUM_SIZE = 24
BIGGER_SIZE = 26

plt.rc('font', size=SMALL_SIZE, weight='bold')  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=17)  # legend fontsize

my_color = "#2a9d8f"
simone_color = "#f4a261"
known_color = "#C26DBC"
mbie_color = "#bc4749"
pessimism_color = "#7b2cbf"

n_runs = 30
p = 1
monitor = "Random", "Ask", "NSupporter", "NExpert", "Level"
# monitor = "Button",

env = (
    # "RiverSwim-6-v0",
    # "Gridworld-Penalty-3x3-v0",
    # "Gridworld-Corridor-3x4-v0",
    "Gridworld-Bottleneck",
)
env_mon_combo = itertools.product(env, monitor)

info = {"RiverSwim-6-v0": {"Ask": (20.02, "optimal"),
                           "Button": (19.14, "optimal"),
                           "Level": (19.83, "optimal"),
                           "N": (20.02, "optimal"),
                           "Random": (20.02, "optimal"),
                           "RandomNonZero": (20.02, "optimal"),
                           "Full": (20.02, "optimal"),
                           },
        "Gridworld-Bottleneck": {"Ask": (0.904, "optimal"),
                                 "Button": (0.19, "optimal"),
                                 "Level": (0.904, "optimal"),
                                 "NSupporter": (0.915, "optimal"),
                                 "NExpert": (0.904, "optimal"),
                                 "Random": (0.904, "optimal"),
                                 "RandomNonZero": (0.904, "optimal"),
                                 "Full": (0.904, "optimal"),
                                 },
        "Gridworld-Corridor-3x4-v0": {"Ask": (0.764, "optimal"),
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
        "Gridworld-Quicksand-Distract-4x4-v0": {"Ask": (0.914, "optimal"),
                                                "Button": (0.821, "optimal"),
                                                "Level": (0.914, "optimal"),
                                                "N": (0.914, "optimal"),
                                                "Random": (0.914, "optimal"),
                                                "RandomNonZero": (0.914, "optimal"),
                                                "Full": (0.914, "optimal"),
                                                },
        }

for env, monitor in env_mon_combo:
    fig, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    algos = [
        (f"{monitor}", known_color, f"{p}" if monitor != "Full" else "1"),
    ]

    assert n_runs == 30

    for conf in algos:
        algo, color, prob = conf
        ref, opt_caut = info[env][monitor]
        my_runs = []
        mbie_runs = []
        # knm_runs = []
        for i in range(n_runs):
            x = np.load(f"data/ablation/mine/solvable/Gym-Grid/"
                        f"{env}/{algo}_{prob}/data_{i}.npz")["test_return"][:500]
            my_runs.append(x)

            x = np.load(
                f"data/ablation/mbie/solvable/Gym-Grid/"
                f"{env}/{algo}_{prob}/data_{i}.npz")["test_return"][:500]
            mbie_runs.append(x)

            # x = np.load(f"data/stochastically_observable/known/"
            #             f"{env}/{algo}_{prob}/data_{i}.npz")["test_return"]
            # knm_runs.append(x)
            # exit()

        # print(np.argmin(np.array(my_runs).sum(-1)))
        # exit()
        my_smoothed = []
        mbie_smoothed = []
        # knm_smoothed = []

        for run in my_runs:
            val = [run[0]]
            for tmp in run[1:]:
                val.append(0.9 * val[-1] + 0.1 * tmp)
            my_smoothed.append(val)

        for run in mbie_runs:
            val = [run[0]]
            for tmp in run[1:]:
                val.append(0.9 * val[-1] + 0.1 * tmp)
            mbie_smoothed.append(val)

        # for run in knm_runs:
        #     val = [run[0]]
        #     for tmp in run[1:]:
        #         val.append(0.9 * val[-1] + 0.1 * tmp)
        #     knm_smoothed.append(val)

        my_mean_return = np.mean(np.asarray(my_smoothed), axis=0)
        my_std_return = np.std(np.asarray(my_smoothed), axis=0)
        my_lower_bound = my_mean_return - 1.96 * my_std_return / math.sqrt(n_runs)
        my_upper_bound = my_mean_return + 1.96 * my_std_return / math.sqrt(n_runs)
        ax.fill_between(np.arange(len(my_mean_return)),
                        my_lower_bound,
                        my_upper_bound,
                        alpha=0.25,
                        color=my_color
                        )
        ax.plot(np.arange(len(my_mean_return)),
                my_mean_return,
                alpha=1,
                linewidth=4,
                c=my_color,
                )

        s_mean_return = np.mean(np.asarray(mbie_smoothed), axis=0)
        s_std_return = np.std(np.asarray(mbie_smoothed), axis=0)
        s_lower_bound = s_mean_return - 1.96 * s_std_return / math.sqrt(n_runs)
        s_upper_bound = s_mean_return + 1.96 * s_std_return / math.sqrt(n_runs)
        ax.fill_between(np.arange(len(s_mean_return)),
                        s_lower_bound,
                        s_upper_bound,
                        alpha=0.25,
                        color=mbie_color
                        )
        ax.plot(np.arange(len(s_mean_return)),
                s_mean_return,
                alpha=1,
                linewidth=4,
                c=mbie_color,
                )

        # knm_mean_return = np.mean(np.asarray(knm_smoothed), axis=0)
        # knm_std_return = np.std(np.asarray(knm_smoothed), axis=0)
        # knm_lower_bound = knm_mean_return - 1.96 * knm_std_return / math.sqrt(n_runs)
        # knm_upper_bound = knm_mean_return + 1.96 * knm_std_return / math.sqrt(n_runs)
        # ax.fill_between(np.arange(len(knm_mean_return)),
        #                 knm_lower_bound,
        #                 knm_upper_bound,
        #                 alpha=0.25,
        #                 color=known_color
        #                 )
        # ax.plot(np.arange(len(knm_mean_return)),
        #         knm_mean_return,
        #         alpha=1,
        #         linewidth=4,
        #         c=known_color,
        #         label="Known Monitor"
        #         )

        plt.axhline(ref, linestyle="--", color="k", linewidth=3, label=f"{opt_caut}")
        # ax.set_ylabel("Discounted Test Return", weight="bold", fontsize=18)
        # ax.legend(loc='lower right', ncol=2, bbox_to_anchor=(1, 0))
        ax.xaxis.set_tick_params(labelsize=20, colors="black")
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 10:.0f}"))
        # plt.title(f"{env}_{monitor}_{prob}")
        # plt.xlabel("Steps (x$10^3$)", weight="bold", fontsize=30)
        ax.xaxis.label.set_color('black')
        ax.set_xticks(np.arange(0, 501, 100))
        # ax.set_xticklabels([])
        ax.set_xlim(0, 500)
        ax.yaxis.set_tick_params(labelsize=20, colors="black")
        ax.set_ylim(0, 1)
        # ax.set_ylim(-0.7, 0.3)
        # ax.set_ylim([-5, 1])

        if monitor == "Button" or monitor == "Random":
        # ax.set_ylabel("Discounted test return",
        #               weight="bold",
        #               fontsize=20,
        #               # rotation="horizontal",
        #               # labelpad=50,
        #               # ha='right'
        #               )
        # ax.legend(loc='lower right', bbox_to_anchor=(1, 0))
        #     ax.set_yticks([-0.5, -0.2, 0.1, 0.3])
            ax.set_yticks([0, 0.2, 0.5, 0.8, 1])
        #     ax.set_yticks([0, -100, -200, -300, -400])
        #     ax.set_yticklabels([])
        #     ax.set_yticks([-4, -2, 0, 1])
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

    # inset Axes....
    # x1, x2, y1, y2 = 0, 300, -0.7, 0.3  # subregion of the original image
    # axins = ax.inset_axes((0.4, 0.62, 0.52, 0.28),
    #                       xlim=(x1, x2), ylim=(y1, y2), xticklabels=[0, 10, 20, 30], yticklabels=[-0.7, 0.3])
    #
    # axins.fill_between(np.arange(len(my_mean_return)),
    #                    my_lower_bound,
    #                    my_upper_bound,
    #                    alpha=0.25,
    #                    color=my_color
    #                    )
    # axins.plot(np.arange(len(my_mean_return)),
    #            my_mean_return,
    #            alpha=1,
    #            linewidth=3,
    #            c=my_color,
    #            )
    # axins.axhline(ref, linestyle="--", color="k", linewidth=2, label=f"{opt_caut}")
    #
    # axins.set_yticks([-0.7, 0.3])
    # axins.set_xticks(np.arange(0, 301, 100))
    # axins.set_xlim(0, 300)
    # axins.xaxis.set_tick_params(labelsize=15, colors="black")
    # axins.yaxis.set_tick_params(labelsize=15, colors="black")
    # ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=2)
    #
    # plt.show()
    plt.savefig(f"/Users/alirezakazemipour/Desktop/{monitor}_{env}_{prob}.pdf",
                format="pdf",
                bbox_inches="tight",
                dpi=300
                )
    plt.close()
