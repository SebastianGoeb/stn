#!/usr/bin/env python3

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from experiment import Experiment


if __name__ == '__main__':
    for i in range(50):
        std_freq = 0.5
        size = 50
        k = 6
        p = 1
        exp = Experiment(
            graph=nx.connected_watts_strogatz_graph(n=size, k=k, p=p),
            freqs=np.random.normal(scale=std_freq, size=size),
            coupling=2,
            max_time=100,
            sync_threshold=1e-7
        )
        exp.run()
        plt.plot(exp.times, exp.std_dphase_dts)
        print(i, 'done')
    plt.yscale('log')
    plt.show()
