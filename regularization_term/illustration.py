import numpy as np
import matplotlib.pyplot as plt

# Grid setup
w1 = np.linspace(-3, 3, 400)
w2 = np.linspace(-3, 3, 400)
W1, W2 = np.meshgrid(w1, w2)

# Real coordinates of the optimal point
a, b = 1.5, 1.0

# Quadratic loss function
loss = (W1 - a) ** 2 + (W2 - b) ** 2

# --- Computation of optimal weights with regularization ---
# For L2 : the point (a,b) is getting closer to the origin following L2 norm
MAGNITUDE_L2 = 1.0

squared_norm_ab = a**2 + b**2

OPTIMUM_RADIUS_FOR_L2 = np.sqrt(squared_norm_ab) - MAGNITUDE_L2
LAMBDA_L2 = np.sqrt(squared_norm_ab / MAGNITUDE_L2**2) - 1

w1_L2 = a / (1 + LAMBDA_L2)
w2_L2 = b / (1 + LAMBDA_L2)

# For L1 : "soft thresholding" effect
MAGNITUDE_L1 = 1.0


def soft_threshold(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)


def lambda_from_magnitude(magnitude, a, b):
    a, b = np.abs(a), np.abs(b)
    if a >= magnitude and b >= magnitude:
        lam = (a + b - magnitude) / 2
    elif a > magnitude and b < magnitude:
        lam = a - magnitude
    elif a < magnitude and b > magnitude:
        lam = b - magnitude
    else:
        lam = 0
    return lam


OPTIMUM_RADIUS_FOR_L1 = np.sqrt(
    2 * (MAGNITUDE_L1 + a - b) ** 2 / 4
    - 2 * (a - b + MAGNITUDE_L1) * (MAGNITUDE_L1 + a - b) / 2
    + a**2
    + (b - MAGNITUDE_L1) ** 2
)
LAMBDA_L1 = lambda_from_magnitude(magnitude=MAGNITUDE_L1, a=a, b=b)

w1_L1 = soft_threshold(a, LAMBDA_L1)
w2_L1 = soft_threshold(b, LAMBDA_L1)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# ===== L2 =====
ax = axs[0]
ax.contour(W1, W2, loss, levels=20, cmap="viridis")
circle = plt.Circle(
    (0, 0), MAGNITUDE_L2, color="red", fill=False, linewidth=2, label="L2 constrain"
)
optimum_circle = plt.Circle(
    (a, b),
    OPTIMUM_RADIUS_FOR_L2,
    color="blue",
    fill=False,
    linewidth=2,
    label="Optimal level",
)
ax.add_artist(circle)
ax.add_artist(optimum_circle)
ax.plot(a, b, "ro", label="Minimum without regularization")
ax.plot(w1_L2, w2_L2, "bo", label="Minimum with L2 regularization")
ax.set_title("L2 Regularization")
ax.set_xlabel("w1")
ax.set_ylabel("w2")
ax.axis("equal")
ax.legend()

# ===== L1 =====
ax = axs[1]
ax.contour(W1, W2, loss, levels=20, cmap="viridis")

c = MAGNITUDE_L1
optimum_circle = plt.Circle(
    (a, b),
    OPTIMUM_RADIUS_FOR_L1,
    color="blue",
    fill=False,
    linewidth=2,
    label="Optimal level",
)
ax.add_artist(optimum_circle)
ax.plot([0, c, 0, -c, 0], [c, 0, -c, 0, c], "r-", linewidth=2, label="L1 constrain")
ax.plot(a, b, "ro", label="Minimum without L1 regularization")
ax.plot(w1_L1, w2_L1, "bo", label="Minimum with L1 regularization")
ax.set_title("L1 Regularization")
ax.set_xlabel("w1")
ax.set_ylabel("w2")
ax.axis("equal")
ax.legend()

plt.suptitle("Effect of L1 and L2 regularization on a given loss function")
plt.tight_layout()
plt.show()

# Affichage des valeurs numériques des solutions
print("Point sans régularisation : (%.2f, %.2f)" % (a, b))
print("Solution L2 régularisée   : (%.2f, %.2f)" % (w1_L2, w2_L2))
print("Solution L1 régularisée   : (%.2f, %.2f)" % (w1_L1, w2_L1))
