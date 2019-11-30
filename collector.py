import pandas as pd
from multiprocessing import Process, Manager
from programsettings import PARALLELISM
import random
import time
import os
import copy
import gc

class Collector:

    def __init__(self, formulation, OutputDirName):
        self.m = formulation.m  # Storing this in the dataset would help in visualizing aggregated traffic volume!
        self.formulation = formulation
        # self.list_of_dict = self.formulation.list_of_dict
        self.num_nodes = formulation.trafficGenerator.topology.num_nodes
        self.num_links = formulation.trafficGenerator.topology.num_links
        self.topo_id = formulation.trafficGenerator.topology.topo_id
        self.obj_val = formulation.obj_val
        self.network_load = formulation.trafficGenerator.network_load
        self.tm_type = formulation.trafficGenerator.tm_type
        self.demandVolumes = self.formulation.demandVolumes

        # fill some known value so far.
        self.dic = {'n': self.num_nodes, 'l': self.num_links, 'k': self.formulation.numCandPath,
                    'obj_val': self.obj_val, 'obj_type': self.formulation.objective,
                    'formulation_type': self.formulation.formulationType, 'network_load': self.network_load,
                    'tm_type': self.tm_type, 'nodal_degree': self.num_links / self.num_nodes,
                    'topo_id': self.topo_id * 10000000000}

        self.calculate_link_utilizations_and_residuals()
        self.extract_flow_value_per_path_index()

        self.calculate_total_residual()

        self.calculate_TE_Metrics()

        # self.list_of_dict.append(self.dic)

        # df.to_csv(filename, index=False)
        df = pd.DataFrame.from_dict([self.dic])
        filename = OutputDirName + '/' + time.strftime("%Y-%m-%d-H%H-M%M-S%S") + "_RND" + str(
            random.random() * 10000000000) + str(os.getpid()) + '.pkl'

        df.to_pickle(filename)

        # self.q.cancel_join_thread()

    def calculate_TE_Metrics(self):
        # Function to calculate some TE Metrics taken from:
        # How Well do Traffic Engineering Objective Functions Meet TE Requirements?
        # Some other metrics from (Comparison of Different QoS-oriented Objectives for
        # Multicommodity Flow Routing Optimization)
        # might be considered as well.

        # u_max => maximum link utilization
        # u_mean => average link utilization
        # ABW_min => minimum available bandwidth
        # ABW_mean => average of available bandwidth
        # l_mean => mean load
        # delay_mean => average network delay

        utilization_list = [self.dic['links_utilization_and_residual'][link]['utilization'] for link in
                            self.dic['links_utilization_and_residual']]
        residuals_list = [self.dic['links_utilization_and_residual'][link]['residual'] for link in
                          self.dic['links_utilization_and_residual']]
        load_list = [self.dic['links_utilization_and_residual'][link]['load'] for link in
                     self.dic['links_utilization_and_residual']]

        capacity_list = [self.dic['links_utilization_and_residual'][link]['capacity'] for link in
                         self.dic['links_utilization_and_residual']]

        u_max = max(utilization_list)
        u_mean = sum(utilization_list) / len(utilization_list)

        ABW_min = min(residuals_list)
        ABW_mean = sum(residuals_list) / len(residuals_list)

        l_mean = sum(load_list) / len(load_list)

        # Now calculating the average delay
        # average delay = sum(l / (c-l))/H where H is the summation of all traffic.
        c_l = [b_i - a_i for b_i, a_i in zip(capacity_list, load_list)]  # (c - l)
        l_over_c_l = [b_i / a_i for b_i, a_i in zip(load_list, c_l)]
        H = sum([self.demandVolumes[s][d] for s in self.demandVolumes for d in self.demandVolumes[s]])
        delay_mean = sum(l_over_c_l) / H

        # Now store them in the dictionary of the dataset
        self.dic['u_max'] = u_max
        self.dic['u_mean'] = u_mean
        self.dic['ABW_min'] = ABW_min
        self.dic['ABW_mean'] = ABW_mean
        self.dic['l_mean'] = l_mean
        self.dic['delay_mean'] = delay_mean

    def calculate_total_residual(self):

        #  calculate summation of residuals
        summ_of_residuals = 0
        summ_of_capacities = 0
        for link in self.dic['links_utilization_and_residual']:
            summ_of_residuals += self.dic['links_utilization_and_residual'][link]['residual']
            summ_of_capacities += self.dic['links_utilization_and_residual'][link]['capacity']
        self.dic['percentage_available_cap'] = (summ_of_residuals * 100) / summ_of_capacities

    def extract_flow_value_per_path_index(self):

        if self.formulation.formulationType == 'singlepath':
            index_to_aggr_flow = {index: 0 for index in range(self.formulation.numCandPath)}
            for v in self.m.getVars():
                if 'Index' in v.varName and 'b' not in v.varName:
                    index = int(v.varName.split('_')[2])
                    index_to_aggr_flow[index] += abs(
                        self.formulation.demandVolumes[self.formulation.RoutesToDemands[v.varName][0]]
                        [self.formulation.RoutesToDemands[v.varName][1]] * self.formulation.bps[v.varName].x
                    )
            # Now store the dictionary in a list ordered from 0 index to the last index.
            self.dic['aggregated_flow_value_per_path_index'] = [index_to_aggr_flow[i] for i in index_to_aggr_flow]

        if self.formulation.formulationType == 'multipath':
            index_to_aggr_flow = {index: 0 for index in range(self.formulation.numCandPath)}
            for v in self.m.getVars():
                if 'Index' in v.varName:
                    index = int(v.varName.split('_')[2])
                    index_to_aggr_flow[index] += v.x
            # Now store the dictionary in a list ordered from 0 index to the last index.
            self.dic['aggregated_flow_value_per_path_index'] = [index_to_aggr_flow[i] for i in index_to_aggr_flow]

    def calculate_link_utilizations_and_residuals(self):
        links_utilization_dic = {}
        for link in self.formulation.linksToRoutes:
            links_utilization_dic[link] = {}
            capacity = self.formulation.topo[link[0]][link[1]]['capacity']
            summ = 0
            for varName in self.formulation.linksToRoutes[link]:
                if self.formulation.formulationType == 'multipath':
                    var = self.formulation.m.getVarByName(varName)
                    summ = summ + var.x
                if self.formulation.formulationType == 'singlepath':
                    var = self.formulation.bps[varName].x * \
                          self.formulation.demandVolumes[self.formulation.RoutesToDemands[varName][0]][
                              self.formulation.RoutesToDemands[varName][1]]
                    summ = summ + var
            links_utilization_dic[link]['capacity'] = capacity
            links_utilization_dic[link]['load'] = summ
            links_utilization_dic[link]['utilization'] = summ / capacity
            links_utilization_dic[link]['residual'] = capacity - summ
            self.dic['links_utilization_and_residual'] = links_utilization_dic
