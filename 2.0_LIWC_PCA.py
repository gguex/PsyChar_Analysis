import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# --- Parameters

# On which book
book_name = "Hamlet"
# Perform PCA on covariance or correlation matrix
PCA_on_cor = True
# Axes to focus on
axes = (0, 1)
# Save results df
save_res_df = False


# --- Loading and preprocessing

# Loading LIWC df
liwc_df = pd.read_csv(f"results/LIWC/LIWC_{book_name}.csv", index_col=0)

# Removing .txt
liwc_df.index = [index_name[:-4] for index_name in liwc_df.index]

# Removing constant columns
liwc_df = liwc_df.loc[:, liwc_df.apply(pd.Series.nunique) > 1]

# Extracting row names, columns name, matrix data, and dimensions
row_names = list(liwc_df.index)
col_names = list(liwc_df.columns)
liwc_matrix = np.array(liwc_df)
n, p = liwc_matrix.shape


# --- PCA

# Centering
X = liwc_matrix - liwc_matrix.mean(axis=0)
# Standardizing (if PCA on cor)
if PCA_on_cor:
    X = X / X.std(axis=0)
# Computation of cov or cor matrix
Co = 1/n * X.T @ X

# Spectral decomposition
val_p, vec_p = np.linalg.eig(Co)
val_p = np.real(val_p)
vec_p = np.real(vec_p)
idx = val_p.argsort()[::-1]
val_p = val_p[idx][:(min(n, p) - 1)]
vec_p = vec_p[:, idx][:, :(min(n, p) - 1)]

# Row factorial scores
F = X @ vec_p
# Percentages of explained variance
p_var = val_p / sum(val_p)
# Saturation
S = np.diag(1 / np.sqrt(np.diag(Co))) @ vec_p @ np.diag(np.sqrt(val_p))
# Communities on selected axis
h = np.sum(S[:, axes] ** 2, axis=1)


# --- Plots

# Row plot
fig, ax = plt.subplots()
# The points
ax.scatter(F[:, axes[0]], F[:, axes[1]], alpha=0.5, s=10)
# Point names
for i, txt in enumerate(row_names):
    ax.annotate(txt, (F[i, axes[0]], F[i, axes[1]]), alpha=0.5, fontsize=8)
# Axes labels
ax.set_xlabel(f"Axis {axes[0] + 1}, {round(p_var[axes[0]] * 100, 2)} %")
ax.set_ylabel(f"Axis {axes[1] + 1}, {round(p_var[axes[1]] * 100, 2)} %")
# Grid
ax.grid()
# Referential
ax.axhline(0, color='black')
ax.axvline(0, color='black')
# Show it
plt.show()

# Col plot
fig, ax = plt.subplots()
# The points
ax.scatter(S[:, axes[0]], S[:, axes[1]], alpha=0.5, s=10)
# Point names
for i, txt in enumerate(col_names):
    ax.annotate(f"{txt}, {round(h[i] * 100)}%", (S[i, axes[0]], S[i, axes[1]]), alpha=0.5, fontsize=8)
# Axes labels
ax.set_xlabel(f"Axis {axes[0] + 1}")
ax.set_ylabel(f"Axis {axes[1] + 1}")
# Grid
ax.grid()
# Referential
ax.axhline(0, color='black')
ax.axvline(0, color='black')
# Scale
plt.xlim((-1, 1))
plt.ylim((-1, 1))
# Circle
circle = plt.Circle((0, 0), 1, fill=False)
ax.add_patch(circle)
# Show it
plt.show()

# --- Results df

# Build row df
row_df = pd.DataFrame(F, index=row_names)
row_df.columns = [f"{i}|{round(p * 100, 2)}%" for i, p in enumerate(p_var)]
# Build col df
col_df = pd.DataFrame(np.round((S ** 2) * 100, 2), index=col_names)

# Save if option is true
if save_res_df:
    row_df.to_csv(f"results/PCA/{book_name}_row.csv")
    col_df.to_csv(f"results/PCA/{book_name}_col.csv")
