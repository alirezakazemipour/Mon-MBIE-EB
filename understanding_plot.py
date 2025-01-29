import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib import ticker

plt.style.use('ggplot')

SMALL_SIZE = 8
MEDIUM_SIZE = 24
BIGGER_SIZE = 26

plt.rc('font', size=SMALL_SIZE, weight='bold')  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=17)  # legend fontsize

n_runs = 30

my_goals = []
my_buttons = []
my_unobsrvs = []

s_goals = []
s_buttons = []
s_unobsrvs = []

for i in range(n_runs):
    x = np.load(f"data/understanding/mine/Gym-Grid/"
                f"Gridworld-Snake-6x6-v0/Button_0.05/data_{i}.npz")

    my_goals.append(x["goal_cnt_hist"])
    my_buttons.append(x["button_off_cnt_hist"] + x["button_on_cnt_hist"])
    my_unobsrvs.append(x["unobsrv_cnt_hist"])

    x = np.load(
        f"data/understanding/simone/iGym-Grid/"
        f"Gridworld-Snake-6x6-v0/ButtonMonitor_0.05/ButtonMonitor__0.05_{i}.npz")

    s_goals.append(x["train/goal_cnt_hist"])
    s_buttons.append(x["train/button_off_cnt_hist"] + x["train/button_on_cnt_hist"])
    s_unobsrvs.append(x["train/unobsrv_cnt_hist"])

my_goals_smoothed = []
my_buttons_smoothed = []
my_unobsrvs_smoothed = []

s_goals_smoothed = []
s_buttons_smoothed = []
s_unobsrvs_smoothed = []

for g, b, u in zip(my_goals, my_buttons, my_unobsrvs):
    val_g, val_b, val_u = [g[0]], [b[0]], [u[0]]

    for tmp_g, tmp_b, tmp_u in zip(g[1:], b[1:], u[1:]):
        val_g.append(0.9 * val_g[-1] + 0.1 * tmp_g)
        val_b.append(0.9 * val_b[-1] + 0.1 * tmp_b)
        val_u.append(0.9 * val_u[-1] + 0.1 * tmp_u)

    my_goals_smoothed.append(val_g)
    my_buttons_smoothed.append(val_b)
    my_unobsrvs_smoothed.append(val_u)

for g, b, u in zip(s_goals, s_buttons, s_unobsrvs):
    val_g, val_b, val_u = [g[0]], [b[0]], [u[0]]

    for tmp_g, tmp_b, tmp_u in zip(g[1:], b[1:], u[1:]):
        val_g.append(0.9 * val_g[-1] + 0.1 * tmp_g)
        val_b.append(0.9 * val_b[-1] + 0.1 * tmp_b)
        val_u.append(0.9 * val_u[-1] + 0.1 * tmp_u)

    s_goals_smoothed.append(val_g)
    s_buttons_smoothed.append(val_b)
    s_unobsrvs_smoothed.append(val_u)

fig, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.set_tick_params(labelsize=20, colors="black")
ax.yaxis.set_tick_params(labelsize=20, colors="black")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 10000:.0f}"))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 10:.0f}"))

my_mean_goals = np.mean(np.asarray(my_goals_smoothed), axis=0)
my_std_goals = np.std(np.asarray(my_goals_smoothed), axis=0)
my_lower_bound = my_mean_goals - 1.96 * my_std_goals / math.sqrt(n_runs)
my_upper_bound = my_mean_goals + 1.96 * my_std_goals / math.sqrt(n_runs)
ax.fill_between(np.arange(len(my_mean_goals)),
                my_lower_bound,
                my_upper_bound,
                alpha=0.25,
                color="blue"
                )
ax.plot(np.arange(len(my_mean_goals)),
        my_mean_goals,
        alpha=1,
        linewidth=4,
        c="blue",
        label="Double MBIE"
        )

s_mean_goals = np.mean(np.asarray(s_goals_smoothed), axis=0)
s_std_goals = np.std(np.asarray(s_goals_smoothed), axis=0)
s_lower_bound = s_mean_goals - 1.96 * s_std_goals / math.sqrt(n_runs)
s_upper_bound = s_mean_goals + 1.96 * s_std_goals / math.sqrt(n_runs)
ax.fill_between(np.arange(len(s_mean_goals)),
                s_lower_bound,
                s_upper_bound,
                alpha=0.25,
                color="red"
                )
ax.plot(np.arange(len(s_mean_goals)),
        s_mean_goals,
        alpha=1,
        linewidth=4,
        c="red",
        label="Directed-E$\mathbf{^2}$"
        )
# plt.yscale('log', base=10)
# plt.ylim([1, 10 ** 7])
# plt.gca().ticklabel_format(useMathText=True)
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
# plt.show()
plt.ylim([-1000, 40000])
ax.set_yticks([0, 10000, 20000, 30000, 40000])
# ax.set_yticklabels([])
plt.savefig(f"/Users/alirezakazemipour/Desktop/Goal_Visited.pdf",
                format="pdf",
                bbox_inches="tight"
                )
# plt.show()
plt.close()
################# Button #########################

fig, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.set_tick_params(labelsize=20, colors="black")
ax.yaxis.set_tick_params(labelsize=20, colors="black")
# ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 10:.0f}"))

my_mean_buttons = np.mean(np.asarray(my_buttons_smoothed), axis=0)
my_std_buttons = np.std(np.asarray(my_buttons_smoothed), axis=0)
my_lower_bound = my_mean_buttons - 1.96 * my_std_buttons / math.sqrt(n_runs)
my_upper_bound = my_mean_buttons + 1.96 * my_std_buttons / math.sqrt(n_runs)
ax.fill_between(np.arange(len(my_mean_buttons)),
                my_lower_bound,
                my_upper_bound,
                alpha=0.25,
                color="blue"
                )
ax.plot(np.arange(len(my_mean_buttons)),
        my_mean_buttons,
        alpha=1,
        linewidth=4,
        c="blue",
        label="Double MBIE"
        )

s_mean_buttons = np.mean(np.asarray(s_buttons_smoothed), axis=0)
s_std_buttons = np.std(np.asarray(s_buttons_smoothed), axis=0)
s_lower_bound = s_mean_buttons - 1.96 * s_std_buttons / math.sqrt(n_runs)
s_upper_bound = s_mean_buttons + 1.96 * s_std_buttons / math.sqrt(n_runs)
ax.fill_between(np.arange(len(s_mean_buttons)),
                s_lower_bound,
                s_upper_bound,
                alpha=0.25,
                color="red"
                )
ax.plot(np.arange(len(s_mean_buttons)),
        s_mean_buttons,
        alpha=1,
        linewidth=4,
        c="red",
        label="Directed-E$\mathbf{^2}$"
        )
# plt.yscale('log',base=10)
# plt.ylim([1, 10 ** 7])
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
ax.set_yticks([0, 10000, 20000, 30000, 40000])

plt.ylim([-1000, 40000])
# plt.show()
ax.set_yticklabels([])
plt.savefig(f"/Users/alirezakazemipour/Desktop/Button_Visited.pdf",
                format="pdf",
                bbox_inches="tight"
                )
plt.close()
################# Unobservs #########################

fig, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.set_tick_params(labelsize=20, colors="black")
ax.yaxis.set_tick_params(labelsize=20, colors="black")
# ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 10:.0f}"))

my_mean_unobsrvs = np.mean(np.asarray(my_unobsrvs_smoothed), axis=0)
my_std_unobsrvs = np.std(np.asarray(my_unobsrvs_smoothed), axis=0)
my_lower_bound = my_mean_unobsrvs - 1.96 * my_std_unobsrvs / math.sqrt(n_runs)
my_upper_bound = my_mean_unobsrvs + 1.96 * my_std_unobsrvs / math.sqrt(n_runs)
ax.fill_between(np.arange(len(my_mean_unobsrvs)),
                my_lower_bound,
                my_upper_bound,
                alpha=0.25,
                color="blue"
                )
ax.plot(np.arange(len(my_mean_unobsrvs)),
        my_mean_unobsrvs,
        alpha=1,
        linewidth=4,
        c="blue",
        label="Double MBIE"
        )

s_mean_unobsrvs = np.mean(np.asarray(s_unobsrvs_smoothed), axis=0)
s_std_unobsrvs = np.std(np.asarray(s_unobsrvs_smoothed), axis=0)
s_lower_bound = s_mean_unobsrvs - 1.96 * s_std_unobsrvs / math.sqrt(n_runs)
s_upper_bound = s_mean_unobsrvs + 1.96 * s_std_unobsrvs / math.sqrt(n_runs)
ax.fill_between(np.arange(len(s_mean_unobsrvs)),
                s_lower_bound,
                s_upper_bound,
                alpha=0.25,
                color="red"
                )
ax.plot(np.arange(len(s_mean_unobsrvs)),
        s_mean_unobsrvs,
        alpha=1,
        linewidth=4,
        c="red",
        label="Directed-E$\mathbf{^2}$"
        )
# plt.yscale('log',base=10)
# plt.ylim([1, 10 ** 7])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
# ax.set_ylabel("Visitation Count",
#               weight="bold",
#               fontsize=20,
#               # rotation="horizontal",
#               # labelpad=50,
#               # ha='right'
#               color="k"
#               )
# plt.title("State-Action Pairs with Unobservable Rewards", weight='bold')
# plt.show()
plt.ylim([-100, 4000])
ax.set_yticks([0, 1000, 2000, 3000, 4000])
# ax.set_yticklabels([])
plt.savefig(f"/Users/alirezakazemipour/Desktop/Unobserved_Visited.pdf",
                format="pdf",
                bbox_inches="tight"
                )
# plt.close()
# plt.show()

# labels = [0, 1000, 2000, 3000, 4000]
# fig, ax = plt.subplots()
# ax.set_ylabel('Area Under the Curve')
# ax.bar(np.arange(5),
#        (my_mean_unobsrvs[0], my_mean_unobsrvs[1000], my_mean_unobsrvs[2000], my_mean_unobsrvs[3000], my_mean_unobsrvs[4000]),
#        width=0.25)
# # ax.errorbar(np.arange(5), (my_mean_unobsrvs[0], my_mean_unobsrvs[1000], my_mean_unobsrvs[2000], my_mean_unobsrvs[3000], my_mean_unobsrvs[4000])
# #             , yerr=(my_std_unobsrvs[0], my_std_unobsrvs[1000], my_std_unobsrvs[2000], my_std_unobsrvs[3000], my_std_unobsrvs[4000])
# #             , fmt="o", color="k")
#
# ax.bar(np.arange(5) + .25,
#        (s_mean_unobsrvs[0], s_mean_unobsrvs[1000], my_mean_unobsrvs[2000], my_mean_unobsrvs[3000], my_mean_unobsrvs[4000]),
#        width=0.25)
# # ax.errorbar(np.arange(5) + 0.75, (s_mean_unobsrvs[0], s_mean_unobsrvs[1000], s_mean_unobsrvs[2000], s_mean_unobsrvs[3000], s_mean_unobsrvs[4000],
# #                                   )
# #             , yerr=(s_std_unobsrvs[0], s_std_unobsrvs[1000], s_std_unobsrvs[2000], s_std_unobsrvs[3000], s_std_unobsrvs[4000])
# #             , fmt="o", color="k")
# plt.xticks(np.arange(5) + 0.75, ["0", "1000", "2000", "3000", "4000"])
# plt.show()