import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Создаем фигуру с двумя подграфиками (axes)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

x = np.linspace(-5, 5, 500)
tanh = np.tanh(x)
sigmoid = 1 / (1 + np.exp(-x))

ax1.plot(x, tanh, color="blue", label="tanh(x)")
ax1.set_title("Гиперболический тангенс (tanh)")
ax1.set_xlabel("x")
ax1.set_ylabel("tanh(x)")
ax1.legend()
ax1.grid(True)

ax2.plot(x, sigmoid, color="red", label="sigmoid(x)")
ax2.set_title("Сигмоида (sigmoid)")
ax2.set_xlabel("x")
ax2.set_ylabel("sigmoid(x)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
