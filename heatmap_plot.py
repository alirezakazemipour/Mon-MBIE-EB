import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

x = np.load(f"data/stochastically_observable/mine/Gym-Grid/"
            f"Gridworld-Empty-Snake-6x6-v0/Button_0.05/data_0.npz"
            )

right_env_reward = x["env_reward_model"][:, 2].reshape(6, 6)
left_env_reward = x["env_reward_model"][:, 0].reshape(6, 6)
stay_env_reward = x["env_reward_model"][:, 4].reshape(6, 6)
env_visit = x["env_visit"].sum(-1).reshape(6, 6)
# right_act_value = x["joint_q"][:, :, :, 0].sum(1).max(1).reshape(6, 6)
# left_act_value = x["joint_q"][:, :, 0, 0].mean(1).reshape(6, 6)
# stay_act_value = x["joint_q"][:, :, 4, 0].mean(1).reshape(6, 6)
# monitor = x["monitor"][:, :, :, :].mean((1, 2, 3)).reshape(6, 6)

# x = stay_env_reward
# sns.heatmap(x, annot=True, fmt=".1f", linewidth=.5, annot_kws={'weight': 'bold'})
# # plt.title("State Visitation")
# plt.axis("off")
# # plt.show()
# plt.savefig(f"/Users/alirezakazemipour/Desktop/stay_env_reward.pdf",
#             bbox_inches="tight"
#             )
# plt.close()
#
# x = right_env_reward
# sns.heatmap(x, annot=True, fmt=".0f", linewidth=.5, annot_kws={'weight': 'bold'})
# plt.title("Double MBIE", weight='bold')
# plt.axis("off")
# # plt.show()
# plt.savefig(f"/Users/alirezakazemipour/Desktop/right_env_reward.pdf",
#             format="pdf",
#             bbox_inches="tight"
#             )
# plt.close()
#
# x = left_env_reward
# sns.heatmap(x, annot=True, fmt=".0f", linewidth=.5, annot_kws={'weight': 'bold'})
# plt.title("Double MBIE", weight='bold')
# plt.axis("off")
# # plt.show()
# plt.savefig(f"/Users/alirezakazemipour/Desktop/left_env_reward.pdf",
#             format="pdf",
#             bbox_inches="tight"
#             )
# plt.close()
#
x = env_visit
sns.heatmap(x, annot=True, fmt=".0f", linewidth=.5, annot_kws={'weight': 'bold'})
plt.title("State Visitation", weight='bold')
plt.axis("off")
# plt.show()
plt.savefig(f"/Users/alirezakazemipour/Desktop/double_mbie_visit_50.pdf",
            bbox_inches="tight",
            format="pdf"
            )
plt.close()

x = np.load(f"data/million_steps/simone/iGym-Grid/"
            f"Gridworld-Empty-Snake-6x6-v0/ButtonMonitor_0.05/ButtonMonitor__0.05_0.npz"
            )
env_visit = x["train/visit_count"].sum((1, 2, 3)).reshape(6, 6)

x = env_visit
sns.heatmap(x, annot=True, fmt=".0f", linewidth=.5, annot_kws={'weight': 'bold'})
plt.title("State Visitation", weight='bold')
plt.axis("off")
# plt.show()
plt.savefig(f"/Users/alirezakazemipour/Desktop/dee_visit_1m.pdf",
            format="pdf",
            bbox_inches="tight"
            )
plt.close()
