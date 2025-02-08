import numpy as np
import heapq


def multinomial_resampling(ws, size=0):
    u = np.random.rand(*ws.shape)
    bins = np.cumsum(ws)
    return np.digitize(u, bins)


def stratified_resampling(ws, size=0):
    # Determine number of elements
    N = len(ws)
    u = (np.arange(N) + np.random.rand(N)) / N
    bins = np.cumsum(ws)
    return np.digitize(u, bins)


def systematic_resampling(ws):
    N = len(ws)

    # make N subdivisions, and choose positions with a consistent random offset
    positions = (np.random.random() + np.arange(N)) / N

    ind = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(ws)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            ind[i] = j
            i += 1
        else:
            j += 1
    return ind


