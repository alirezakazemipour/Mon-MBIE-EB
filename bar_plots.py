import matplotlib.pyplot as plt
import math
import numpy as np

n_runs = 100
algos = [
    "PartialObsButton_1",
    "PartialObsButton_0.75",
    "PartialObsButton_0.5",
    "PartialObsButton_0.25",
    "PartialObsButton_0.1"
]
meta_runs = []
meta_bounds = []
for algo in algos:
    runs = []
    for i in range(n_runs):
        x = np.load(f"data/mine/{algo}/test_{i}.npy")
        runs.append(np.trapz(x, dx=100))

    # smoothed = []
    # for run in runs:
    #     val = [run[0]]
    #     for tmp in run[1:]:
    #         val.append(0.9 * val[-1] + 0.1 * tmp)
    #     smoothed.append(val)
    mean_auc = np.mean(np.asarray(runs), axis=0)
    meta_runs.append(mean_auc)
    std_return = np.std(np.asarray(runs), axis=0)
    bound = 1.96 * std_return / math.sqrt(n_runs)
    meta_bounds.append(bound)

x = np.arange(len(algos))  # the label locations
width = 0.5  # the width of the bars
multiplier = 0

labels = ["100%", "75%", "50%", "25%", "10%"]
fig, ax = plt.subplots()
ax.set_ylabel('Area Under the Curve')
ax.bar(np.arange(len(algos)), meta_runs)
ax.errorbar(np.arange(len(algos)), meta_runs, yerr=meta_bounds, fmt="o", color="r")
ax.set_xticks(x+ width, labels)
ax.set_yscale("log")
plt.show()
