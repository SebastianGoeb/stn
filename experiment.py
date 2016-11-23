
import numpy as np
import networkx as nx
from time import time


class Experiment(object):
    def __init__(self, graph, freqs, coupling, max_time, sync_threshold):
        # Constants
        self.dt = (2 * np.pi) / (10 * coupling)  # 10 steps per rotation for largest phase adjustment (=coupling)
        self.max_time = max_time
        self.sync_threshold = sync_threshold
        self.coupling = coupling
        self.graph = graph
        self.natural_frequencies = freqs
        adj = nx.adjacency_matrix(self.graph).toarray()
        self.multiplier = adj * coupling / (np.sum(adj, 1) + 1)[:, None]

        # Runtime properties
        self.runtime = 0
        self.time = 0
        self.phases = np.random.rand(nx.number_of_nodes(self.graph)) * 2 * np.pi

        # Data
        self.std_dphase_dts = None
        self.times = None

    def update(self):
        phase_diffs = self.phases[None, :] - self.phases[:, None]
        phase_adjustments = np.sum(np.sin(phase_diffs) * self.multiplier, 1)
        dphase_dts = self.natural_frequencies + phase_adjustments
        self.phases += dphase_dts * self.dt
        self.time += self.dt
        return dphase_dts

    def run(self):
        start_time = time()
        self.time = 0
        self.times = None
        while self.time < self.max_time:
            dphase_dts = self.update()
            if self.times is None:
                self.std_dphase_dts = np.std(dphase_dts)
                self.times = np.array(self.time)
            else:
                self.std_dphase_dts = np.append(self.std_dphase_dts, np.std(dphase_dts))
                self.times = np.append(self.times, self.time)
            if np.std(dphase_dts) < self.sync_threshold:
                break
        self.runtime = time() - start_time
