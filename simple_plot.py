import matplotlib.pyplot as plt
import math
import numpy as np

n_runs = 30
algos = [
    # "MDP",
    ("Button_1.0", "blue", "100%"),
    ("Button_0.75", "red", "75%"),
    ("Button_0.5", "green", "50%"),
    ("Button_0.25", "orange", "25%"),
    ("Button_0.1", "brown", "10%"),
    ("Button_0.01", "magenta", "1%")
]
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(6.4, 4.8))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

SMALL_SIZE = 8
MEDIUM_SIZE = 24
BIGGER_SIZE = 26

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=17)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)


for conf in algos:
    algo, color, legend = conf
    runs = []
    for i in range(n_runs):
        x = np.load(f"data/Gym-Grid/RiverSwim-6-v0/{algo}/test_{i}.npy")
        runs.append(x)

    smoothed = []
    for run in runs:
        val = [run[0]]
        for tmp in run[1:]:
            val.append(0.9 * val[-1] + 0.1 * tmp)
        smoothed.append(val)
    mean_return = np.mean(np.asarray(smoothed), axis=0)
    std_return = np.std(np.asarray(smoothed), axis=0)
    lower_bound = mean_return - 1.96 * std_return / math.sqrt(n_runs)
    upper_bound = mean_return + 1.96 * std_return / math.sqrt(n_runs)
    ax.fill_between(np.arange(len(mean_return)),
                    lower_bound,
                    upper_bound,
                    alpha=0.25,
                    color=color
                    )
    ax.plot(np.arange(len(mean_return)),
            mean_return,
            alpha=1,
            linewidth=3,
            c=color,
            label=legend
            )
plt.axhline(-151.88, linestyle="--", color="k", linewidth=3, label="optimal")
ax.set_ylabel("Discounted Test Return", weight="bold", fontsize=18)
plt.title(f"EOP", weight="bold")
ax.legend(loc="upper left")
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
# for i in range(25, 30):
#     ax.plot(np.arange(len(mean_return)),
#             smoothed[i],
#             alpha=1,
#             linewidth=3,
#             c=np.random.rand(3,)
#             )

plt.show()
# plt.savefig("/Users/alirezakazemipour/Desktop/ask_grid.pdf",
#             format="pdf",
#             bbox_inches="tight"
#             )

