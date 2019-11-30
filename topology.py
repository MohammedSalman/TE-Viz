import networkx as nx
import time
from config import NETWORK_LOAD, CAPACITY_SET, CAPACITY_TYPE, TOPOLOGY_SEED,\
    FIXED_WEIGHT_VALUE, WEIGHT_SETTING, Nu_of_TMs_Per_TOPO, TM_TYPES
import fnss as fnss
import matplotlib.pyplot as plt
from programsettings import VISUALIZE_TOPOLOGY, WAITING_TIME, PARALLELISM, CPU_COUNT
from trafficgenerator import TrafficGenerator
from multiprocessing import Process
from gurobipy import *
import random
from itertools import product




class Topology:
    """
    Stores network topology graph, configures capacities, visualize graph.
    """

    def __init__(self, num_nodes, num_links, OutputDirName):
        # self.q = q
        self.num_nodes = num_nodes
        self.num_links = num_links
        self.topo_id = random.random()
        self.topo = self.create_topology(num_nodes, num_links, TOPOLOGY_SEED)


        if self.topo is not None:
            self.cap_type = CAPACITY_TYPE
            self.cap_set = CAPACITY_SET
            if self.cap_type == 'edge_betweenness':
                self.set_capacities()
                self.set_weights()
            if VISUALIZE_TOPOLOGY:
                self.visualizeGraph('spring', node_size=175, show_edge_labels=True)

            if isinstance(NETWORK_LOAD, str):
                LOAD = [float(NETWORK_LOAD)]
            elif isinstance(NETWORK_LOAD, list):
                LOAD = NETWORK_LOAD
            else:
                raise ValueError('UNKNOWN Traffic Matrix load')

            if isinstance(TM_TYPES, str):
                TM_TYPES_ = [float(NETWORK_LOAD)]
            elif isinstance(TM_TYPES, list):
                TM_TYPES_= TM_TYPES
            else:
                raise ValueError('UNKNOWN Traffic Matrix type')
            # print("topo: ", self.topo.edges(data=True))

            if PARALLELISM:
                procs = []

                for network_load, R, tm_type in product(LOAD, range(Nu_of_TMs_Per_TOPO), TM_TYPES_):
                    proc = Process(target=TrafficGenerator, args=(network_load, self, tm_type, OutputDirName))
                    procs.append(proc)

                for i in self.chunks(procs, CPU_COUNT):
                    for j in i:
                        j.start()
                    for j in i:
                        j.join()

            else:
                objects = [TrafficGenerator(network_load, self, TM_TYPES_, OutputDirName) for network_load, R in
                           product(LOAD, range(Nu_of_TMs_Per_TOPO))]
    @staticmethod
    def chunks(ll, nn):
        for index in range(0, len(ll), nn):
            yield ll[index:index + nn]

    def set_capacities(self):
        fnss.set_capacities_edge_betweenness(self.topo, self.cap_set, 'Gbps')
        # set_capacities_betweenness_gravity(g, capacity_set, 'Gbps')
        # set_capacities_communicability_gravity(g, capacity_set, 'Gbps')

    def set_weights(self):
        if WEIGHT_SETTING == 'fixed':
            fnss.set_weights_constant(self.topo, FIXED_WEIGHT_VALUE)
        if WEIGHT_SETTING == 'inv-cap':
            fnss.set_weights_inverse_capacity(self.topo)

    @staticmethod
    def create_topology(n, l, TOPOLOGY_SEED):
        printed = False

        if l <= n > 10:
            return None
        if l > ((n * (n - 1)) / 2):
            return None
        tic = time.time()
        while True:
            g = nx.dense_gnm_random_graph(n, l, seed=TOPOLOGY_SEED)
            if nx.is_connected(g):
                if printed is True:
                    logging.warning('Found a connected graph n = %d, l = %d', n, l)
                return g.to_directed()
            if printed is False:
                logging.warning('Disconnected graph, trying another graph! n = %d, l = %d', n, l)
                printed = True
            if time.time() - tic > WAITING_TIME:
                logging.warning('Could not find a connected graph n = %d, l = %d', n, l)
                exit(0)
                # return None

    def visualizeGraph(self, layout, node_size=150, iterations=100, show_edge_labels=False):
        g = self.topo
        if layout == 'circular':
            pos = nx.circular_layout(g)
        elif layout == 'shell':
            pos = nx.shell_layout(g)
        elif layout == 'spring':
            pos = nx.spring_layout(g, iterations=iterations)
        elif layout == 'fruchterman':
            pos = nx.fruchterman_reingold_layout(g, iterations=iterations)
        elif layout == 'kamada':
            pos = nx.kamada_kawai_layout(g)
        elif layout == 'spectral':
            pos = nx.spectral_layout(g)
        elif layout == 'rescale':
            pass
            # pos = nx.rescale_layout(g)  # it needs a shape
        elif layout == 'included_in_graph':
            pos = nx.get_node_attributes(g, 'pos')
        elif layout == 'grid':
            pos = dict(zip(g, g))  # nice idea of drawing a grid
        else:
            pos = nx.random_layout(g)

        # nx.draw_networkx_nodes(g, pos, cmap=plt.get_cmap('jet'), node_size=node_size)
        # nx.draw_networkx_labels(g, pos)
        # nx.draw_networkx_edges(g, pos, arrows=True)
        # nx.draw_networkx_edges(g, pos, arrows=False)

        plt.figure(figsize=(8, 8))
        # plt.subplot(111)
        nx.draw_networkx_edges(g, pos, alpha=0.5, arrows=False, width=3)
        nx.draw_networkx_nodes(g, pos,
                               node_size=700,
                               cmap=plt.cm.Reds_r)

        # plt.xlim(-0.05, 1.05)
        # plt.ylim(-0.05, 1.05)
        plt.axis('off')
        nx.draw_networkx_labels(g, pos)
        if show_edge_labels == True:
            nx.draw_networkx_edge_labels(g, pos, font_size=7)
        '''draw_networkx_edge_labels(g, pos,
                                  edge_labels=None,
                                  label_pos=0.5,
                                  font_size=10,
                                  font_color='k',
                                  font_family='sans-serif',
                                  font_weight='normal',
                                  alpha=1.0,
                                  bbox=None,
                                  ax=None,
                                  rotate=True,
                                  **kwds):'''

        # filename = time.strftime("%Y%m%d-%H%M%S") + '_NetGraph' + ImageFileExtention
        # plt.savefig('svg/' + filename)

        plt.show()
