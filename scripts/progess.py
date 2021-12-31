import csv
import sys
import numpy as np
import matplotlib.pyplot as plt


data = np.genfromtxt(sys.argv[1], dtype=float, skip_header=0, delimiter=',', names=True)
print(data.dtype)

epoch_idx    = data['epoch']
batch_idx    = data['batch']
batch_loss   = data['loss']
epoch_loss   = data['epoch_loss']
average_loss = data['exponential_moving_average_loss']

batch_size   = max(batch_idx) + 1

batch_idx    = [batch_idx[i] + batch_size * epoch_idx[i] for i in range(len(batch_idx))]

plt.plot(batch_idx, batch_loss)
plt.plot(batch_idx, epoch_loss)
plt.plot(batch_idx, average_loss)

plt.show()