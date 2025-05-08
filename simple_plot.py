import matplotlib.pyplot as plt
import math
import numpy as np
import itertools
from matplotlib import ticker
import os

mon_mbie_eb_color = "#2a9d8f"  # noqa
de2_color = "#f4a261"

n_runs = 30
p = 1, 0.8, 0.2, 0.05
monitor = "Random", "Ask", "NSupporter", "NExpert", "Level", "Button"
env = "Gridworld-Bottleneck",

env_mon_p_combo = itertools.product(env, monitor, p)

info = {"RiverSwim-6-v0": {"Ask": (20.02, "Minimax-Optimal"),
                           "Button": (19.14, "Minimax-Optimal"),
                           "Level": (19.83, "Minimax-Optimal"),
                           "N": (20.02, "Minimax-Optimal"),
                           "Random": (20.02, "Minimax-Optimal"),
                           "RandomNonZero": (20.02, "Minimax-Optimal"),
                           "Full": (20.02, "Minimax-Optimal"),
                           },
        "Gridworld-Bottleneck": {"Ask": (0.904, "Minimax-Optimal"),
                                 "Button": (0.19, "Minimax-Optimal"),
                                 "Level": (0.904, "Minimax-Optimal"),
                                 "NSupporter": (0.915, "Minimax-Optimal"),
                                 "NExpert": (0.904, "Minimax-Optimal"),
                                 "Random": (0.904, "Minimax-Optimal"),
                                 "RandomNonZero": (0.904, "Minimax-Optimal"),
                                 "Full": (0.904, "Minimax-Optimal"),
                                 },
        "Gridworld-Corridor-3x4-v0": {"Ask": (0.764, "Minimax-Optimal"),
                                      "Button": (0.672, "Minimax-Optimal"),
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
        "Gridworld-Quicksand-Distract-4x4-v0": {"Ask": (0.914, "Minimax-Optimal"),
                                                "Button": (0.821, "Minimax-Optimal"),
                                                "Level": (0.914, "Minimax-Optimal"),
                                                "N": (0.914, "Minimax-Optimal"),
                                                "Random": (0.914, "Minimax-Optimal"),
                                                "RandomNonZero": (0.914, "Minimax-Optimal"),
                                                "Full": (0.914, "Minimax-Optimal"),
                                                },
        }

assert n_runs == 30

for env, monitor, prob in env_mon_p_combo:
    fig, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ref, ref_label = info[env][monitor]
    mon_mbie_eb_runs = []
    dee_runs = []
    knm_runs = []
    for i in range(n_runs):
        x = np.load(f"data/Mon_MBIE_EB/{env}/{monitor}_{prob}/data_{i}.npz")["test_return"]
        mon_mbie_eb_runs.append(x)
        x = np.load(f"data/DEE/{env}/{monitor}_{prob}/data_{i}.npz")["test/return"]
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
    plt.title(f"{env}_{monitor}({prob * 100}%)")
    plt.xlabel("Training Steps (x$10^3$)")

    ax.set_xticks(np.arange(0, 501, 100))

    if monitor != "Button":
        ax.set_yticks([0, 0.2, 0.5, 0.8, 1])
        ax.set_ylim([0, 1])
    else:
        ax.set_yticks([-0.5, -0.2, 0.1, 0.3])
        ax.set_ylim([-0.7, 0.3])

    os.makedirs("figs", exist_ok=True)
    plt.savefig(f"figs/{env}_{monitor}({prob * 100}%).pdf",
                format="pdf",
                bbox_inches="tight",
                dpi=300
                )
    plt.close()
