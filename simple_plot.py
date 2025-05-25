import matplotlib.pyplot as plt
import math
import numpy as np
import itertools
from matplotlib import ticker
import os
import warnings

de2_exists = True
if not os.path.exists("data/DE2"):
    de2_exists = False
    warnings.warn("Directed-E$^2$ data not found. Ignoring it.")
    
mon_mbie_eb_exists = True
if not os.path.exists("data/Mon_MBIE_EB"):
    mon_mbie_eb_exists = False
    warnings.warn("Mon_MBIE_EB data not found. Ignoring it.")

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
                                 "MDP": (0.904, "Minimax-Optimal"),
                                 },
        }

assert n_runs == 30

for env, monitor, prob in env_mon_p_combo:
    fig, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ref, ref_label = info[env][monitor]
    mon_mbie_eb_runs = []
    de2_runs = []
    knm_runs = []
    for i in range(n_runs):
        if mon_mbie_eb_exists:
            x = np.load(f"data/Mon_MBIE_EB/{env}/{monitor}_{prob}/data_{i}.npz")["test_return"]
            mon_mbie_eb_runs.append(x)
        if de2_exists:
            x = np.load(f"data/DE2/{env}/{monitor}_{prob}/data_{i}.npz")["test/return"]
            de2_runs.append(x)
        x = np.load(f"data/Known_Monitor/{env}/{monitor}_{prob}/data_{i}.npz")["test_return"]
        knm_runs.append(x)

    mon_mbie_eb_smoothed = []
    de2_smoothed = []
    knm_smoothed = []

    if mon_mbie_eb_exists:
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

    for run in knm_runs:
        val = [run[0]]
        for tmp in run[1:]:
            val.append(0.9 * val[-1] + 0.1 * tmp)
        knm_smoothed.append(val)

    if mon_mbie_eb_exists:
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