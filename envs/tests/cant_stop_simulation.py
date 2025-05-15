import numpy as np
import time

n = int(5e6)

start = time.time()
all = np.random.randint(np.ones((n, 11), dtype=np.uint8) + (np.array([3,5,7,9,11,13,11,9,7,5,3], dtype=np.uint8)))
mask = (all == 0).astype(np.uint32)
counts = np.sum(mask, axis=1)

n_0 = np.sum((counts == 0).astype(int))
n_1 = np.sum((counts == 1).astype(int))
n_2 = np.sum((counts == 2).astype(int))

n_012 = n_0 + n_1 + n_2
print(n_012 / n)
print(n_0 / n_012)
print(n_1 / n_012)
print(n_2 / n_012)
print("time", time.time() - start)
del all

start = time.time()
n = 1
m = 11
vals = np.array([3, 5, 7, 9, 11, 13, 11, 9, 7, 5, 3], dtype=np.uint8)

# Use the modern NumPy random generator for better performance
rng = np.random.default_rng()
all_ = rng.integers(1, vals + 1, size=(n, m), dtype=np.uint8)
k_values = rng.integers(0, 3, size=n, dtype=np.uint8)
row_indices = np.repeat(np.arange(n, dtype=np.int32), k_values)

# Generate random values for sorting (using float32 for memory efficiency)
rand_vals = rng.random(size=(n, m), dtype=np.float32)
sorted_indices = np.argsort(rand_vals, axis=1, kind='stable')
mask = np.arange(m, dtype=np.uint8) < k_values[:, None]
col_indices = sorted_indices[mask]
all_[row_indices, col_indices] = 0
print(all_)
quit()
# Calculate counts of zeros per row without intermediate arrays
counts = (all_ == 0).sum(axis=1, dtype=np.uint8)

# Compute the required statistics
n_0 = (counts == 0).sum()
n_1 = (counts == 1).sum()
n_2 = (counts == 2).sum()

n_012 = n_0 + n_1 + n_2

# Output the results
print(n_012 / n)
print(n_0 / n_012)
print(n_1 / n_012)
print(n_2 / n_012)
print("time", time.time() - start)