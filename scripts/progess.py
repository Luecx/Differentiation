import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
import time

# plt.show(block=False)

# f = plt.Figure()
# graph = f.add_subplot(111)


def plot():
    data = np.genfromtxt(sys.argv[1], dtype=float, skip_header=0, delimiter=',', names=True)

    print("reading", flush=True)

    epoch_idx = data['epoch']
    batch_idx = data['batch']
    batch_loss = data['loss']
    epoch_loss = data['epoch_loss']
    average_loss = data['exponential_moving_average_loss']

    batch_size = max(batch_idx) + 1

    batch_idx = [batch_idx[i] + batch_size * epoch_idx[i] for i in range(len(batch_idx))]

    # graph.clear()

    plt.plot(batch_idx, batch_loss, c="r")
    plt.plot(batch_idx, epoch_loss, c="b")
    plt.plot(batch_idx, average_loss, c="g")



pause_time = -1
if len(sys.argv) >= 3:
    pause_time = int(sys.argv[2])

if pause_time > 0:
    while(True):
        plot()
        plt.pause(pause_time)
else:
    plot()
    plt.show()

