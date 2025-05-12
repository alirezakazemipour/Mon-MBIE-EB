import matplotlib.pyplot as plt
import math
import numpy as np
import itertools
from matplotlib import ticker
import os

mon_mbie_eb_color = "#2a9d8f"  # noqa
optim_mbie_eb_color = "#bc4749"

n_runs = 30
p = 1,
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
    optim_mbie_eb_runs = []
    knm_runs = []
    for i in range(n_runs):
        x = np.load(f"data/Mon_MBIE_EB/{env}/{monitor}_{prob}/data_{i}.npz")["test_return"]
        mon_mbie_eb_runs.append(x)
        x = np.load(f"data/Optim_MBIE_EB/{env}/{monitor}_{prob}/data_{i}.npz")["test_return"]
        optim_mbie_eb_runs.append(x)

    mon_mbie_eb_smoothed = []
    optim_mbie_eb_smoothed = []

    for run in mon_mbie_eb_runs:
        val = [run[0]]
        for tmp in run[1:]:
            val.append(0.9 * val[-1] + 0.1 * tmp)
        mon_mbie_eb_smoothed.append(val)

    for run in optim_mbie_eb_runs:
        val = [run[0]]
        for tmp in run[1:]:
            val.append(0.9 * val[-1] + 0.1 * tmp)
        optim_mbie_eb_smoothed.append(val)

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

    optim_mbie_eb_mean_return = np.mean(np.asarray(optim_mbie_eb_smoothed), axis=0)
    optim_mbie_eb_std_return = np.std(np.asarray(optim_mbie_eb_smoothed), axis=0)
    optim_mbie_eb_lower_bound = optim_mbie_eb_mean_return - 1.96 * optim_mbie_eb_std_return / math.sqrt(n_runs)
    optim_mbie_eb_upper_bound = optim_mbie_eb_mean_return + 1.96 * optim_mbie_eb_std_return / math.sqrt(n_runs)
    ax.fill_between(np.arange(len(optim_mbie_eb_mean_return)),
                    optim_mbie_eb_lower_bound,
                    optim_mbie_eb_upper_bound,
                    alpha=0.25,
                    color=optim_mbie_eb_color
                    )
    ax.plot(np.arange(len(optim_mbie_eb_mean_return)),
            optim_mbie_eb_mean_return,
            alpha=1,
            linewidth=4,
            c=optim_mbie_eb_color,
            label="Optimistic MBIE-EB"
            )

    plt.axhline(ref, linestyle="--", color="k", linewidth=3, label=f"{ref_label}")
    ax.set_ylabel("Discounted Test Return")
    ax.legend(loc='upper right', ncol=2, bbox_to_anchor=(1, 0.5))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 10:.0f}"))
    plt.title(f"{env}_{monitor}({prob * 100}%)")
    plt.xlabel("Training Steps (x$10^3$)")

    ax.set_xlim([0, 50])
    ax.set_xticks(np.arange(0, 501, 100))

    ax.set_yticks([-400, -300, -200, -100, 0])
    ax.set_ylim([-420, 5])

    os.makedirs("figs", exist_ok=True)
    plt.savefig(f"figs/{env}_{monitor}({prob * 100}%).pdf",
                format="pdf",
                bbox_inches="tight",
                dpi=300
                )
    plt.close()
