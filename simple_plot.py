import matplotlib.pyplot as plt
import math
import numpy as np

algos = ["PartialObsButton_0.1_cedar", "PartialObsButton_0.1"]#, "PartialObsButton_0.1", "PartialObsButton_0.01"]
for algo in algos:
    runs = []
    for i in range(30):
        x = np.load(f"data/Gym-Monitor/RiverSwim-6-v0/{algo}/test_{i}.npy")
        runs.append(x)
    # print(np.argmin(np.asarray(runs).sum(-1)))
    # exit()
    smoothed = []
    for run in runs:
        val = [run[0]]
        for tmp in run[1:]:
            val.append(0.9 * val[-1] + 0.1 * tmp)
        smoothed.append(val)
    mean_return = np.mean(np.asarray(smoothed), axis=0)
    std_return = np.std(np.asarray(smoothed), axis=0)
    lower_bound = mean_return - 1.96 * std_return / math.sqrt(len(runs))
    upper_bound = mean_return + 1.96 * std_return / math.sqrt(len(runs))
    plt.fill_between(np.arange(len(mean_return)),
                     lower_bound,
                     upper_bound,
                     alpha=0.25
                     )
    plt.plot(np.arange(len(mean_return)),
             mean_return,
             alpha=1,
             label=algo,
             linewidth=3
             )
# plt.fill_between(np.arange(len(mean_return)),
#                  20 - 4.5,
#                  20 + 4.5,
#                  alpha=0.15,
#                  color="magenta"
#                  )
plt.axhline(19.11, linestyle='--', label="optimal", c="magenta")
# plt.axhline(0.941, linestyle='--', label="cautious", c="olive")
plt.xlabel("training steps (x 100)")
plt.ylabel("discounted test return")
plt.title(f" performance over {len(runs)} runs")
plt.grid()
plt.legend()
# for i in range(30):
#     plt.plot(np.arange(len(mean_return)),
#              smoothed[i],
#              label=algo,
#              linewidth=3
#              )
plt.show()

