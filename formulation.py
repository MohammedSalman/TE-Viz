from gurobipy import *
from itertools import islice, combinations
import networkx as nx
from programsettings import MIPGapAbs, GUROBILogToConsole, PRINT_LOG
import pandas as pd
from collector import Collector
import os
import time


class Formulation:
    def __init__(self, formulationType, numCandPath, objective, trafficGenerator, OutputDirName):
        self.trafficGenerator = trafficGenerator
        # self.q = q
        try:
            self.m = Model(name="model")
        except:
            print("Gurobi licence error")
            exit(0)
        self.obj_val = 0
        # This will disable console output
        self.m.params.LogToConsole = GUROBILogToConsole
        self.m.params.MIPGapAbs = MIPGapAbs
        self.topo = trafficGenerator.topo
        self.demandVolumes = trafficGenerator.tm
        self.formulationType = formulationType
        self.numCandPath = numCandPath  # Number of candidate path(k)
        self.objective = objective
        self.feasible = True

        self.pathsVariables = {}  # These are the decision variables
        self.bps = {}  # These are the binary variables
        self.demandsToRoutes = {}
        self.RoutesToDemands = {}  # One to one dictionary

        # pathCost: dictionary of dictionaries
        # outside keys are (s, d) tuples,
        # inside keys are 'routes', values are summed up cost for each route.
        self.pathsCost = {}
        self.linksToRoutes = {}  # Mapping a link to all routes passing that link
        for link in self.topo.edges():
            self.linksToRoutes[link] = []

        self.all_pairs = list(combinations(self.topo.nodes(), 2))  # ingress, egress pairs

        for pair in self.all_pairs:
            s = pair[0]
            d = pair[1]
            self.demandsToRoutes[(s, d)] = []
            self.demandsToRoutes[(d, s)] = []
            self.pathsCost[(s, d)] = {}
            self.pathsCost[(d, s)] = {}

        self.formulate_paths_flow_var()

        # Optimization
        if objective == 'LB' and formulationType == 'multipath':
            self.formulate_LB_Multipath()
        if objective == 'LB' and formulationType == 'singlepath':
            self.formulate_LB_Singlepath()
        if objective == 'MCR' and formulationType == 'multipath':
            self.formulate_MCR_Multipath()
        if objective == 'MCR' and formulationType == 'singlepath':
            self.formulate_MCR_Singlepath()
        if objective == 'AD' and formulationType == 'multipath':
            self.formulate_AD_Multipath()
        if objective == 'AD' and formulationType == 'singlepath':
            self.formulate_AD_Singlepath()

        self.solve()
        if not self.feasible:
            return
        Collector(self, OutputDirName)

    def solve(self):
        # self.m.write("Mymodel.lp")  # TODO: this should have a unique name to the object, overwritten otherwise.
        # self.m.write("Mymodel.mps")
        self.m.optimize()

        # for v in self.m.getVars():
        #     print(v.varName, ": ", v.x)
        try:
            ObjValue = float('%.8f' % self.m.getObjective().getValue())
        except AttributeError:
            print("Model infeasible")
            self.feasible = False
            return

        if PRINT_LOG:

            print("Objective value: ", ObjValue, " num nodes: ", self.trafficGenerator.topology.num_nodes,
                  " num links: ", self.trafficGenerator.topology.num_links,
                  " routing strategy: ", self.formulationType,
                  " network_load: ", self.trafficGenerator.network_load,
                  " obj_type: ", self.objective,
                  " candidate_paths(K): ", self.numCandPath
                  )
        self.obj_val = ObjValue

    def formulate_MCR_Multipath(self):
        self.m.setObjective(quicksum(self.pathsCost[sd_tuples][var] * self.pathsVariables[var]
                                     for sd_tuples in self.pathsCost
                                     for var in self.pathsCost[sd_tuples]),
                            GRB.MINIMIZE)
        self.addDemandConstr()
        # Adding capacity constraints:
        for link in self.linksToRoutes:
            tmp_list = []
            for variable in self.linksToRoutes[link]:
                tmp_list.append(self.pathsVariables[variable])
            self.m.addConstr(sum(tmp_list) <= self.topo[link[0]][link[1]]['capacity'])
        self.m.update()

    def formulate_MCR_Singlepath(self):
        self.m.setObjective(quicksum(self.pathsCost[sd_tuples][var] * self.bps[var] *
                                     self.demandVolumes[self.RoutesToDemands[var][0]][
                                         self.RoutesToDemands[var][1]]
                                     for sd_tuples in self.pathsCost
                                     for var in self.pathsCost[sd_tuples]),
                            GRB.MINIMIZE)

        for demand in self.demandsToRoutes:
            # pass
            self.m.addConstr(quicksum(self.bps[variableName] for variableName in self.demandsToRoutes[demand]) == 1)

        # Adding capacity constraints:
        for link in self.linksToRoutes:
            self.m.addConstr(quicksum(
                self.bps[variable] * self.demandVolumes[self.RoutesToDemands[variable][0]][
                    self.RoutesToDemands[variable][1]]
                for variable in self.linksToRoutes[link]) <= self.topo[link[0]][link[1]]['capacity'])
        self.m.update()

    def formulate_LB_Singlepath(self):
        # page 147. 2018 ebook
        self.z = self.m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="z")
        self.m.setObjective(self.z, GRB.MINIMIZE)

        for demand in self.demandsToRoutes:
            # pass
            self.m.addConstr(quicksum(self.bps[variableName] for variableName in self.demandsToRoutes[demand]) == 1)

        for link in self.linksToRoutes:
            self.m.addConstr(quicksum(
                self.bps[variable] * self.demandVolumes[self.RoutesToDemands[variable][0]][
                    self.RoutesToDemands[variable][1]]
                for variable in self.linksToRoutes[link]) <= self.z * self.topo[link[0]][link[1]]['capacity'])
            # ============ Adding explicit capacity constraint ==============
            # self.m.addConstr(quicksum(  # same as the one above but just removed self.z.
            #     self.bps[variable] * self.demandVolumes[self.RoutesToDemands[variable][0]][
            #         self.RoutesToDemands[variable][1]]
            #     # this constrain might make the problem tight.
            #     for variable in self.linksToRoutes[link]) <= self.topo[link[0]][link[1]]['capacity'])

        self.m.update()

    def formulate_LB_Multipath(self):
        # one more variable for the objective function:
        self.z = self.m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="z")
        self.m.setObjective(self.z, GRB.MINIMIZE)
        self.addDemandConstr()
        self.addCapacityConstr()
        self.m.update()

    def formulate_AD_Multipath(self):
        # See the 2018 book, Page 136
        # TODO: try the piece-wise builtin feature in Gurobi.
        '''for link in linksToRoutes:
            for variable in linksToRoutes[link]:
                r = m.addVar(name='c' + str(link[0]) + '_' + str(link[1]))'''
        # r = m.addVars(linksToRoutes.keys())

        r = self.m.addVars(self.linksToRoutes.keys(), name="r")

        # print("r: ", r)
        c = {}
        for link in self.linksToRoutes:
            c[link] = self.topo[link[0]][link[1]]['capacity']
        # print("c: ", c)
        self.m.setObjective(quicksum(r[link] / c[link] for link in self.linksToRoutes), GRB.MINIMIZE)
        self.addDemandConstr()

        y = self.m.addVars(self.linksToRoutes.keys())
        # print("y: ", y)
        for link in self.linksToRoutes:
            tmp_list = tuplelist()
            for variable in self.linksToRoutes[link]:
                tmp_list.append(self.pathsVariables[variable])
            self.m.addConstr((quicksum(tmp_list)) == y[link])

        '''m.addConstrs(
            (quicksum(nutritionValues[f, c] * buy[f] for f in foods)
             == [minNutrition[c], maxNutrition[c]]
             for c in categories), "_")'''
        # print(type(r), type(y))
        self.m.update()

        for link in self.linksToRoutes:
            self.m.addConstr(r[link] >= 3 / 2 * y[link])
            self.m.addConstr(r[link] >= 9 / 2 * y[link] - c[link])
            self.m.addConstr(r[link] >= 15 * y[link] - 8 * c[link])
            self.m.addConstr(r[link] >= 50 * y[link] - 36 * c[link])
            self.m.addConstr(r[link] >= 200 * y[link] - 171 * c[link])
            self.m.addConstr(r[link] >= 4000 * y[link] - 3781 * c[link])
            self.m.update()

    def formulate_AD_Singlepath(self):

        # TODO: try the piece-wise builtin feature in Gurobi.
        '''for link in linksToRoutes:
            for variable in linksToRoutes[link]:
                r = m.addVar(name='c' + str(link[0]) + '_' + str(link[1]))'''
        # r = m.addVars(linksToRoutes.keys())

        r = self.m.addVars(self.linksToRoutes.keys(), name="r")

        # print("r: ", r)
        c = {}
        for link in self.linksToRoutes:
            c[link] = self.topo[link[0]][link[1]]['capacity']
        # print("c: ", c)
        self.m.setObjective(quicksum(r[link] / c[link] for link in self.linksToRoutes), GRB.MINIMIZE)

        # self.addDemandConstr()
        for demand in self.demandsToRoutes:
            # pass
            self.m.addConstr(quicksum(self.bps[variableName] for variableName in self.demandsToRoutes[demand]) == 1)

        y = self.m.addVars(self.linksToRoutes.keys())
        # print("y: ", y)
        for link in self.linksToRoutes:
            tmp_list = tuplelist()
            for variable in self.linksToRoutes[link]:
                tmp_list.append(self.demandVolumes[self.RoutesToDemands[variable][0]][
                                    self.RoutesToDemands[variable][1]] * self.bps[variable])
            self.m.addConstr((quicksum(tmp_list)) == y[link])

        '''m.addConstrs(
            (quicksum(nutritionValues[f, c] * buy[f] for f in foods)
             == [minNutrition[c], maxNutrition[c]]
             for c in categories), "_")'''
        # print(type(r), type(y))

        for link in self.linksToRoutes:
            self.m.addConstr(r[link] >= 3 / 2 * y[link])
            self.m.addConstr(r[link] >= 9 / 2 * y[link] - c[link])
            self.m.addConstr(r[link] >= 15 * y[link] - 8 * c[link])
            self.m.addConstr(r[link] >= 50 * y[link] - 36 * c[link])
            self.m.addConstr(r[link] >= 200 * y[link] - 171 * c[link])
            self.m.addConstr(r[link] >= 4000 * y[link] - 3781 * c[link])

        self.m.update()
    def addDemandConstr(self):
        for demand in self.demandsToRoutes:
            tmp_list = []
            for variable in self.demandsToRoutes[demand]:
                tmp_list.append(self.pathsVariables[variable])
            self.m.addConstr(sum(tmp_list) == self.demandVolumes[demand[0]][demand[1]])
        self.m.update()

    def addCapacityConstr(self):
        # Adding the constraints for:
        # summation of all traffic for paths using link L divided by L's capacity should not exceed 'z'
        for link in self.linksToRoutes:
            tmp_list = []
            for variable in self.linksToRoutes[link]:
                tmp_list.append(self.pathsVariables[variable])
            self.m.addConstr(sum(tmp_list) <= self.z * self.topo[link[0]][link[1]]['capacity'])
            # m.addConstr(sum(tmp_list) <= g[link[0]][link[1]]['capacity'])  # '''REMOVING THIS CONSTR WILL AFFECT EFFICIENCY'''
        self.m.update()

    def k_shortest_paths(self, source, target):
        return list(islice(nx.shortest_simple_paths(self.topo, source, target, weight='weight'), self.numCandPath))

    def formulate_paths_flow_var(self):
        for pair in self.all_pairs:
            s = pair[0]
            d = pair[1]
            for path_n, path in enumerate(self.k_shortest_paths(s, d)):
                for i in range(2):  # len(pair)
                    if i == 1:
                        path = path[::-1]  # Reverse path, so we don't need to run k_shortest_path again!
                    variableName = 'x' + "_Index_" + str(path_n) + '_s' + str(s) + '_d' + str(d)
                    self.pathsCost[(s, d)][variableName] = 0
                    for vi in range(len(path) - 1):
                        self.pathsCost[(s, d)][variableName] += self.topo[path[vi]][path[vi + 1]]['weight']
                    for index in range(len(path) - 1):
                        self.linksToRoutes[(path[index], path[index + 1])].append(variableName)
                    self.demandsToRoutes[(s, d)].append(variableName)
                    self.pathsVariables[variableName] = self.m.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                                                      name=variableName)
                    self.m.addConstr(self.pathsVariables[variableName] >= 0.0)
                    if self.formulationType == 'singlepath':
                        self.bps[variableName] = self.m.addVar(vtype=GRB.BINARY, name='b_' + str(variableName))
                        self.RoutesToDemands[variableName] = (s, d)
                    s, d = d, s  # Swapping
        self.m.update()
