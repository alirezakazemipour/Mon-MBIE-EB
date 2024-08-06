import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


x = np.load(f"heatmap/Gym-Grid/Gridworld-TwoRoom-Quicksand-3x5-v0/Button_0.5/values_84.npy")
x = x[:, 0].reshape(3, 5)
sns.heatmap(x, annot=True, fmt=".0f", linewidth=.5)
plt.show()
print(x.shape)
