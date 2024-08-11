import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

x = np.load(f"heatmap/Gym-Grid/Gridworld-Penalty-3x3-v0/Ask_0.6/data_30.npz")
obsrv_q = x["obsrv_q"]
joint_count = x["joint_count"]
joint_obsrv_count = x["joint_obsv_count"]
monitor = x["monitor"]
env_obsrv_count = x["env_obsrv_count"]
env_reward_model = x["env_reward_model"]

x = obsrv_q[:, 0, :, 0].mean(-1).reshape(3, 3)
sns.heatmap(x, annot=True, fmt=".1f", linewidth=.5)
plt.title("average observability value when not asking in each state of the environment")


# plt.show()
plt.savefig(f"/Users/alirezakazemipour/github/ofu/trials/penalty_ask_0.6_beta_0.1_notask_obsrv_value.jpg",
            bbox_inches="tight"
            )
