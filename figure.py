import numpy as np
import matplotlib.pyplot as plt

# Create a smooth curve
x = np.linspace(0, 10, 400)
y = 5 * (x / 10)**0.7  # smooth concave curve

# Pick a point for local geometry
i = 200
x0, y0 = x[i], y[i]
slope = np.gradient(y, x)[i]
alpha = np.arctan(slope)

# Small segment for ds
dx = x[1] - x[0]
ds = np.sqrt(1 + slope**2) * dx

# Plot
fig, ax = plt.subplots(figsize=(8, 5))

# Main curve
ax.plot(x, y, linewidth=2.5)

# Points A and B
ax.scatter([0, 10], [0, 5], s=60)
ax.text(0, 0, "  A", fontsize=12)
ax.text(10, 5, "  B", fontsize=12)

# Highlight small segment ds
ax.plot(x[i-3:i+3], y[i-3:i+3], linewidth=4)
ax.text(x0+0.3, y0+0.2, r"$ds$", fontsize=12)

# Tangent line
xt = np.array([x0-1.5, x0+1.5])
yt = y0 + slope*(xt - x0)
ax.plot(xt, yt, linestyle="--", linewidth=1.8)
ax.text(x0-1.8, y0-0.3, "tangent", fontsize=10)

# Angle alpha
theta = np.linspace(0, alpha, 100)
r = 0.8
ax.plot(x0 + r*np.cos(theta), y0 + r*np.sin(theta), linewidth=1.2)
ax.text(x0 + 0.9, y0 + 0.2, r"$\alpha$", fontsize=12)

# Label slope
ax.text(x0-2.5, y0-0.6, r"$\tan\alpha = y'$", fontsize=11)

# Labels
ax.set_xlabel("$x$")
ax.set_ylabel("$y$ (downward)")
ax.set_title("Geometry of the Sliding Path")

# Clean style
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(alpha=0.2)

plt.tight_layout()

# Save for poster
plt.savefig("figures/geometry_concept_poster.png", dpi=300)

plt.show()