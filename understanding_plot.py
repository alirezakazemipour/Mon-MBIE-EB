import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

x = np.load(f"debug/Gym-Grid/"
            f"Gridworld-Snake-6x6-v0/Button_0.05/data_0.npz"
            )

g = x["goal_cnt_hist"]
plt.plot(np.arange(len(g)), g)
b = x["button_cnt_hist"]
plt.plot(np.arange(len(b)), b, "--")
u = x["unobsrv_cnt_hist"]
plt.plot(np.arange(len(u)), u, "red")
plt.show()