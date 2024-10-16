import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

x = np.load(f"data/mine/Gym-Grid/Gridworld-Quicksand-Distract-4x4-v0/Ask_1.0/data_0.npz")
visits = x["env_visit"]

x = visits.mean(-1).reshape(4, 4)
sns.heatmap(x, annot=True, fmt=".1f", linewidth=.5)
plt.title("monitor value for the up action")


plt.show()
# plt.savefig(f"/Users/alirezakazemipour/github/ofu/trials/penalty_ask_0.6_beta_0.1_q_obsrv_max_60_monitor_up_action.jpg",
#             bbox_inches="tight"
#             )
plt.close()
