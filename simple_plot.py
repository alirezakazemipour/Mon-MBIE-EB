import matplotlib.pyplot as plt
import math
import numpy as np
import itertools
from matplotlib import ticker
import os

mon_mbie_eb_color = "#2a9d8f"  # noqa
de2_color = "#f4a261"
known_monitor_color = "#c26dbc"

n_runs = 30
p = 1, 0.8, 0.2, 0.05
monitor = "FullRandom", "Ask", "NSupporters", "NExperts", "Level", "Button"
env = "Gridworld-Bottleneck",

env_mon_p_combo = itertools.product(env, monitor, p)

info = {
        "Gridworld-Bottleneck": {"Ask": (0.904, "Minimax-Optimal"),
                                 "Button": (0.19, "Minimax-Optimal"),
                                 "Level": (0.904, "Minimax-Optimal"),
                                 "NSupporters": (0.915, "Minimax-Optimal"),
                                 "NExperts": (0.904, "Minimax-Optimal"),
                                 "FullRandom": (0.904, "Minimax-Optimal"),
                                 "SemiRandom": (0.904, "Minimax-Optimal"),
                                 "Full": (0.904, "Minimax-Optimal"),
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
        x = np.load(f"data/Known_Monitor/{env}/{monitor}_{prob}/data_{i}.npz")["test_return"]
        knm_runs.append(x)

    mon_mbie_eb_smoothed = []
    dee_smoothed = []
    knm_smoothed = []

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

    for run in knm_runs:
        val = [run[0]]
        for tmp in run[1:]:
            val.append(0.9 * val[-1] + 0.1 * tmp)
        knm_smoothed.append(val)

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

    knm_mean_return = np.mean(np.asarray(knm_smoothed), axis=0)
    knm_std_return = np.std(np.asarray(knm_smoothed), axis=0)
    knm_lower_bound = knm_mean_return - 1.96 * knm_std_return / math.sqrt(n_runs)
    knm_upper_bound = knm_mean_return + 1.96 * knm_std_return / math.sqrt(n_runs)
    ax.fill_between(np.arange(len(knm_mean_return)),
                    knm_lower_bound,
                    knm_upper_bound,
                    alpha=0.25,
                    color=known_monitor_color
                    )
    ax.plot(np.arange(len(knm_mean_return)),
            knm_mean_return,
            alpha=1,
            linewidth=4,
            c=known_monitor_color,
            label="Known Monitor"
            )

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