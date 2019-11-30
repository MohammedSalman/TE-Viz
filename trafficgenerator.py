import numpy.random as nr
import config
from formulation import Formulation
from programsettings import PARALLELISM, PRINT_TM, CPU_COUNT
from multiprocessing import Process
from itertools import product
from config import OBJECTIVES, ROUTING_STRATEGIES, CANDIDATE_PATHS
from fnss import *
import os


class TrafficGenerator:
    """
        Generates different types of traffic matrices: Gravity, Bimodal ... etc.
    """

    def __init__(self, network_load, topology, tm_type, OutputDirName):
        self.topology = topology
        self.network_load = network_load
        self.tm_type = tm_type
        # self.q = q
        self.topo = topology.topo
        self.tm = {}
        for s in self.topo.nodes():
            self.tm[s] = {}
            for d in self.topo.nodes():
                self.tm[s][d] = 0.0

        if self.tm_type == 'gravity':
            self.generateGravityTM()
        if self.tm_type == 'bimodal':
            self.generateBimodalTM()
        if self.tm_type == 'nucci':
            self.generateNucciTM()

        if PRINT_TM:
            print(self.tm)

        if isinstance(ROUTING_STRATEGIES, str):
            routing_strategies = [ROUTING_STRATEGIES]
        else:
            assert isinstance(ROUTING_STRATEGIES, list)
            routing_strategies = ROUTING_STRATEGIES

        if isinstance(CANDIDATE_PATHS, int):
            candidate_paths = [CANDIDATE_PATHS]
        else:
            assert isinstance(CANDIDATE_PATHS, list)
            candidate_paths = CANDIDATE_PATHS

        if isinstance(OBJECTIVES, str):
            objectives = [OBJECTIVES]
        else:
            assert isinstance(OBJECTIVES, list)
            objectives = OBJECTIVES

        if PARALLELISM:

            procs = []
            for formulationType, numCandPath, objective in product(routing_strategies, candidate_paths, objectives):
                proc = Process(target=Formulation, args=(formulationType, numCandPath, objective, self, OutputDirName))
                procs.append(proc)

            for i in self.chunks(procs, CPU_COUNT):
                for j in i:
                    j.start()
                for j in i:
                    j.join()


        else:
            objects = [Formulation(formulationType, numCandPath, objective, self, OutputDirName)
                       for formulationType, numCandPath, objective in
                       product(routing_strategies, candidate_paths, objectives)]

    @staticmethod
    def chunks(ll, nn):
        for index in range(0, len(ll), nn):
            yield ll[index:index + nn]

    def generateGravityTM(self):
        Cap = {}
        totalcap = 0
        for link in self.topo.edges():
            Cap[link[0]] = 0
        for link in self.topo.edges():
            Cap[link[0]] += self.topo[link[0]][link[1]]['capacity']
            totalcap += self.topo[link[0]][link[1]]['capacity']
        for src in self.tm:
            outgoing = Cap[src]
            cap = totalcap - Cap[src]
            for dst in self.tm:
                if dst != src:
                    self.tm[src][dst] = self.network_load * outgoing * Cap[dst] / cap

    def generateBimodalTM(self):
        Cap = {}
        totalcap = 0
        for src in self.topo:
            cap = 0
            for dst in self.topo[src]:
                cap += self.topo[src][dst]['capacity']
            Cap[src] = cap
            totalcap += cap

        for src in self.tm:
            outgoing = Cap[src]
            SIG = 20.0 / (150 * 150)
            for dst in self.tm:
                if src == dst:
                    self.tm[src][dst] = 0
                    continue
                if nr.random() < 0.2:
                    FRAC = 1.67 * self.network_load
                else:
                    FRAC = 0.7 * self.network_load
                self.tm[src][dst] = nr.normal(1, SIG) * FRAC * outgoing / (float(len(self.tm[src])))

    def generateNucciTM(self):
        demandVolumes = static_traffic_matrix(self.topo, mean=0.5, stddev=0.5, max_u=self.network_load)
        self.tm = demandVolumes.flow


# todo: UNUSED FUNCTION! WILL BE REVISITED AFTER REACHING THE VISUALIZAITON PHASE.
def generate_demands(g, **kwargs):
    if kwargs['type'] == 'gravity':
        # FRAC_UTIL = 0.15  # Avg utilization of a link to generate TM#
        Cap = {}
        totalcap = 0
        # topo = g.edges(data=True)
        # print("topo: ", type(topo))

        '''for src in topo:
            cap = 0
            for dst in topo[src]:
                #cap += topo[src][dst]['cap']
                cap += g[src][dst]['capacity']
            Cap[src] = cap
            totalcap += cap'''

        for link in g.edges():
            Cap[link[0]] = 0
        for link in g.edges():
            Cap[link[0]] += g[link[0]][link[1]]['capacity']
            totalcap += g[link[0]][link[1]]['capacity']

        # fill tm first with all 0's
        tm = {}
        for src in g.nodes():
            tm[src] = {}
            for dst in g.nodes():
                if src == dst:
                    continue
                tm[src][dst] = 0

        for src in tm:
            outgoing = Cap[src]
            cap = totalcap - Cap[src]
            for dst in tm:
                if dst != src:
                    tm[src][dst] = FRAC_UTIL * outgoing * Cap[dst] / cap

        # print("GRAVITY TRAFFIC MATRIX:")
        return (tm)

    if kwargs['type'] == 'random':
        demandVolumes = {}
        # Building random dictionary of demands:
        # print(g.nodes)
        for s in g.nodes():
            demandVolumes[s] = {}
            for d in g.nodes():
                if s == d:
                    continue
                '''rand01 = np.random.randint(2)
                print(rand01)
                if rand01 == 1:
                    demandVolumes[(s, d)] = 0
                    continue #making some demands as 0'''
                demandVolumes[s][d] = np.random.randint(low=0, high=5)
                # demandVolumes[s][d] = 0
        # demandVolumes[0][2] = 5
        # demandVolumes[2][0] = 3

        # demandVolumes[(s, d)] = 0
        return demandVolumes

    if kwargs['type'] == 'fnss':
        demandVolumes = static_traffic_matrix(g, mean=0.5, stddev=0.09, max_u=kwargs['max_u'],
                                              seed=kwargs['seed'])
        return demandVolumes.flow
    # print("here they are: ", demandVolumes)
    if kwargs['type'] == 'modified_fnss':
        demandVolumes = trafficmatrixGenerator.modified_static_traffic_matrix(g, mean=0.5,
                                                                              stddev=0.09,
                                                                              max_u=kwargs['max_u'],
                                                                              load_percentage=kwargs[
                                                                                  'load_percentage'],
                                                                              seed=kwargs['seed'])
        '''#pair that do not exist, add it, and assign it to 0
        for s in g.nodes():
            for d in g.nodes():
                if s == d:
                    continue
                if s not in demandVolumes.flow.keys() or d not in demandVolumes[s].flow.keys():
                #if demandVolumes.flow[s][d] not in demandVolumes.flow:
                    demandVolumes.flow[s][d] = 0'''

        return demandVolumes.flow
