import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

x = np.load(f"heatmap/Gym-Grid/Gridworld-Penalty-3x3-v0/Ask_0.6/data_30.npz")
obsrv_q = x["obsrv_q"]
joint_count = x["joint_count"]
joint_obsrv_count = x["joint_obsv_count"]
monitor = x["monitor"]
env_obsrv_count = x["env_obsrv_count"]

x = monitor[:, 0, :, 1].mean(-1).reshape(3, 3)
sns.heatmap(x, annot=True, fmt=".1f", linewidth=.5)
plt.title("average observability for each action in each state of the environment")

# plt.figure()
# y = np.load(f"heatmap/Gym-Grid/Gridworld-TwoRoom-Quicksand-3x5-v0/Button_0.51/visits_84.npy")
# y = y.sum(-1).reshape(3, 5)
# sns.heatmap(y, annot=True, fmt=".0f", linewidth=.5)


# plt.show()
plt.savefig(f"/Users/alirezakazemipour/github/ofu/trials/penalty_ask_0.6_beta_0.01.jpg",
            bbox_inches="tight"
            )
