# topology settings
# N = [i for i in range(80, 100, 5)]
# L = [i for i in range(125, 300, 5)]

N = [12]
L = [27]

Nu_of_TOPOs_Per_N_L = 1
Nu_of_TMs_Per_TOPO = 36

TOPOLOGY_SEED = None  # None, or any int # 448672
CAPACITY_TYPE = 'edge_betweenness'
CAPACITY_SET = [30, 35, 40]

WEIGHT_SETTING = 'inv-cap'  # 'fixed', 'inv-cap'
FIXED_WEIGHT_VALUE = 1  # has no effect if WEIGHT_SETTING is not 'fixed'

TM_TYPES = ['gravity']  # 'gravity' , 'nucci' , 'bimodal'
NETWORK_LOAD = [0.1]

OBJECTIVES = ['LB']
# CANDIDATE_PATHS = [i for i in range(1, 8)]
CANDIDATE_PATHS = [1]
ROUTING_STRATEGIES = ['singlepath']  # ['singlepath', 'multipath']
# VISUALIZATION
DRAW_AGGREGATED_TRAFFIC = False
DRAW_UTILIZATION = False
DRAW_minK_MaxK_DEVIATION = False