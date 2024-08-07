import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


x = np.load(f"heatmap/Gym-Grid/Gridworld-TwoRoom-Quicksand-3x5-v0/Button_0.6/visits_84.npy")
x = x.sum(-1).reshape(3, 5)
sns.heatmap(x, annot=True, fmt=".0f", linewidth=.5)

plt.figure()
y = np.load(f"heatmap/Gym-Grid/Gridworld-TwoRoom-Quicksand-3x5-v0/Button_0.51/visits_84.npy")
y = y.sum(-1).reshape(3, 5)
sns.heatmap(y, annot=True, fmt=".0f", linewidth=.5)



plt.show()
print(x.shape)
