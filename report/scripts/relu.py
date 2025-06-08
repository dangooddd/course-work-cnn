import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

fig, ax = plt.subplots()

x = np.linspace(-5, 5, 500)
relu = np.maximum(0, x)
ax.plot(x, relu, color="blue", label="ReLU(x) = max(0, x)")
ax.set_xlabel("x")
ax.set_ylabel("ReLU(x)")
ax.grid(True)

plt.tight_layout()
plt.show()
