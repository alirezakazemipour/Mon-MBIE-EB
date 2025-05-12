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
                             "RandomNonZero": (0.904, "Minimax-Optimal"),
                             "MDP": (0.904, "Minimax-Optimal"),
                             },
}

assert n_runs == 30

for env, monitor, prob in env_mon_p_combo:
    _, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ref, ref_label = info[env][monitor]

    mon_mbie_eb_runs = []
    mon_mbie_eb_unobsrvs = []
    mon_mbie_eb_goals = []

    dee_runs = []
    dee_unobsrvs = []
    dee_goals = []

    for i in range(n_runs):
        x = np.load(f"data/Mon_MBIE_EB/{env}/{monitor}_{prob}/data_{i}.npz")
        mon_mbie_eb_runs.append(x["test_return"])
        mon_mbie_eb_unobsrvs.append(x["unobsrv_cnt_hist"])
        mon_mbie_eb_goals.append(x["goal_cnt_hist"])

        x = np.load(f"data/DEE/{env}/{monitor}_{prob}/data_{i}.npz")
        dee_runs.append(x["test/return"])
        dee_goals.append(x["test/goal_cnt_hist"])
        dee_unobsrvs.append(x["test/unobsrv_cnt_hist"])

    mon_mbie_eb_smoothed = []
    mon_mbie_eb_goal_smoothed = []
    mon_mbie_eb_unobsrv_smoothed = []

    dee_smoothed = []
    dee_goal_smoothed = []
    dee_unobsrv_smoothed = []

    for run, goals, unobsrvs in zip(mon_mbie_eb_runs, mon_mbie_eb_goals, mon_mbie_eb_unobsrvs):
        val, goal, unobsrv = [run[0]], [goals[0]], [unobsrvs[0]]

        for tmp_run, tmp_goal, tmp_unobsrv in zip(run[1:], goals[1:], unobsrvs[1:]):
            val.append(0.9 * val[-1] + 0.1 * tmp_run)
            goal.append(0.9 * goal[-1] + 0.1 * tmp_goal)
            unobsrv.append(0.9 * unobsrv[-1] + 0.1 * tmp_unobsrv)
        mon_mbie_eb_smoothed.append(val)
        mon_mbie_eb_goal_smoothed.append(goal)
        mon_mbie_eb_unobsrv_smoothed.append(unobsrv)

    for run, goals, unobsrvs in zip(dee_runs, dee_goals, dee_unobsrvs):
        val, goal, unobsrv = [run[0]], [goals[0]], [unobsrvs[0]]

        for tmp_run, tmp_goal, tmp_unobsrv in zip(run[1:], goals[1:], unobsrvs[1:]):
            val.append(0.9 * val[-1] + 0.1 * tmp_run)
            goal.append(0.9 * goal[-1] + 0.1 * tmp_goal)
            unobsrv.append(0.9 * unobsrv[-1] + 0.1 * tmp_unobsrv)
        dee_smoothed.append(val)
        dee_goal_smoothed.append(goal)
        dee_unobsrv_smoothed.append(unobsrv)

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
    plt.savefig(f"figs/Return_{env}_{monitor}({prob * 100}%).pdf",
                format="pdf",
                bbox_inches="tight",
                dpi=300
                )

    # Goals
    _, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    mon_mbie_eb_mean_goals = np.mean(np.asarray(mon_mbie_eb_goal_smoothed), axis=0)
    mon_mbie_eb_std_goals = np.std(np.asarray(mon_mbie_eb_goal_smoothed), axis=0)
    mon_mbie_eb_lower_bound = mon_mbie_eb_mean_goals - 1.96 * mon_mbie_eb_std_goals / math.sqrt(n_runs)
    mon_mbie_eb_upper_bound = mon_mbie_eb_mean_goals + 1.96 * mon_mbie_eb_std_goals / math.sqrt(n_runs)
    ax.fill_between(np.arange(len(mon_mbie_eb_mean_goals)),
                    mon_mbie_eb_lower_bound,
                    mon_mbie_eb_upper_bound,
                    alpha=0.25,
                    color=mon_mbie_eb_color
                    )
    ax.plot(np.arange(len(mon_mbie_eb_mean_goals)),
            mon_mbie_eb_mean_goals,
            alpha=1,
            linewidth=4,
            c=mon_mbie_eb_color,
            label="Mon-MBIE-EB"
            )

    dee_mean_goals = np.mean(np.asarray(dee_goal_smoothed), axis=0)
    dee_std_goals = np.std(np.asarray(dee_goal_smoothed), axis=0)
    dee_lower_bound = dee_mean_goals - 1.96 * dee_std_goals / math.sqrt(n_runs)
    dee_upper_bound = dee_mean_goals + 1.96 * dee_std_goals / math.sqrt(n_runs)
    ax.fill_between(np.arange(len(dee_mean_goals)),
                    dee_lower_bound,
                    dee_upper_bound,
                    alpha=0.25,
                    color=de2_color
                    )
    ax.plot(np.arange(len(dee_mean_goals)),
            dee_mean_goals,
            alpha=1,
            linewidth=4,
            c=de2_color,
            label="Directed-E$\mathbf{^2}$"  # noqa
            )
    ax.set_ylabel("Goal Visitation Count")
    ax.legend(loc='lower right', ncol=2, bbox_to_anchor=(1, 0))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 10:.0f}"))
    plt.title(f"{env}_{monitor}({prob * 100}%)")
    plt.xlabel("Training Steps (x$10^3$)")

    ax.set_xlim([0, 500])
    ax.set_xticks(np.arange(0, 501, 100))

    plt.savefig(f"figs/Goal_visits_{env}_{monitor}({prob * 100}%).pdf",
                format="pdf",
                bbox_inches="tight",
                dpi=300
                )
    plt.close()

    # Unobservs # noqa

    _, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    mon_mbie_eb_mean_unobsrvs = np.mean(np.asarray(mon_mbie_eb_unobsrv_smoothed), axis=0)
    mon_mbie_eb_std_unobsrvs = np.std(np.asarray(mon_mbie_eb_unobsrv_smoothed), axis=0)
    mon_mbie_eb_lower_bound = mon_mbie_eb_mean_unobsrvs - 1.96 * mon_mbie_eb_std_unobsrvs / math.sqrt(n_runs)
    mon_mbie_eb_upper_bound = mon_mbie_eb_mean_unobsrvs + 1.96 * mon_mbie_eb_std_unobsrvs / math.sqrt(n_runs)
    ax.fill_between(np.arange(len(mon_mbie_eb_mean_unobsrvs)),
                    mon_mbie_eb_lower_bound,
                    mon_mbie_eb_upper_bound,
                    alpha=0.25,
                    color=mon_mbie_eb_color
                    )
    ax.plot(np.arange(len(mon_mbie_eb_mean_unobsrvs)),
            mon_mbie_eb_mean_unobsrvs,
            alpha=1,
            linewidth=4,
            c=mon_mbie_eb_color,
            label="Mon-MBIE-EB"
            )

    dee_mean_unobsrvs = np.mean(np.asarray(dee_unobsrv_smoothed), axis=0)
    dee_std_unobsrvs = np.std(np.asarray(dee_unobsrv_smoothed), axis=0)
    dee_lower_bound = dee_mean_unobsrvs - 1.96 * dee_std_unobsrvs / math.sqrt(n_runs)
    dee_upper_bound = dee_mean_unobsrvs + 1.96 * dee_std_unobsrvs / math.sqrt(n_runs)
    ax.fill_between(np.arange(len(dee_mean_unobsrvs)),
                    dee_lower_bound,
                    dee_upper_bound,
                    alpha=0.25,
                    color=de2_color
                    )
    ax.plot(np.arange(len(dee_mean_unobsrvs)),
            dee_mean_unobsrvs,
            alpha=1,
            linewidth=4,
            c=de2_color,
            label="Directed-E$\mathbf{^2}$"  # noqa
            )
    ax.set_ylabel("Unobserved Visitation Count")
    ax.legend(loc='lower right', ncol=2, bbox_to_anchor=(1, 0))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 10:.0f}"))
    plt.title(f"{env}_{monitor}({prob * 100}%)")
    plt.xlabel("Training Steps (x$10^3$)")
    ax.set_xticks(np.arange(0, 501, 100))
    plt.savefig(f"figs/Unobsrv_visits_{env}_{monitor}({prob * 100}%).pdf",
                format="pdf",
                bbox_inches="tight",
                dpi=300
                )
    plt.close()
