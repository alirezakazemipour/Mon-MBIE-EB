import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


x = np.load(f"heatmap/Gym-Grid/Gridworld-Empty-Distract-6x6-v0/Button_0.1/values_84.npy")
x = x[:, 2].reshape(6, 6)
sns.heatmap(x, annot=True, fmt=".0f", linewidth=.5)
plt.show()
print(x.shape)
