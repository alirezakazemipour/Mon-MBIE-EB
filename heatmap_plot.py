import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


x = np.load(f"heatmap/Gym-Grid/Gridworld-Corridor-3x4-v0/Button_0.25/visits_84.npy")
x = (x[:, 2].reshape(3, 4))
sns.heatmap(x, annot=True, fmt=".1f", linewidth=.5)

# plt.figure()
# y = np.load(f"heatmap/Gym-Grid/Gridworld-TwoRoom-Quicksand-3x5-v0/Button_0.51/visits_84.npy")
# y = y.sum(-1).reshape(3, 5)
# sns.heatmap(y, annot=True, fmt=".0f", linewidth=.5)



plt.show()
print(x.shape)
