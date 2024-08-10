import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


x = np.load(f"heatmap/Gym-Grid/Gridworld-Penalty-3x3-v0/Button_0.4/data_387.npz")
obsrv_q = x["obsrv_q"]
joint_count = x["joint_count"]
joint_obsrv_count = x["joint_obsv_count"]

obsrv_q_off = obsrv_q[:, 0, ...]
obsrv_q_on = obsrv_q[:, 1, ...]


x = joint_count[:, 1,...].sum(axis=(-2)).reshape(3, 3)
sns.heatmap(x, annot=True, fmt=".1f", linewidth=.5)

# plt.figure()
# y = np.load(f"heatmap/Gym-Grid/Gridworld-TwoRoom-Quicksand-3x5-v0/Button_0.51/visits_84.npy")
# y = y.sum(-1).reshape(3, 5)
# sns.heatmap(y, annot=True, fmt=".0f", linewidth=.5)



plt.show()
