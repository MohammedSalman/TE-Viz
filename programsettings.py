import os

CPU_COUNT = os.cpu_count() // 4
PARALLELISM = True
VISUALIZE_TOPOLOGY = False
WAITING_TIME = 100  # How long to wait when making a graph before returning None.
MIPGapAbs = 0.1  # was 0.02
GUROBILogToConsole = 0  # 0 or 1
PRINT_LOG = 1
PRINT_TM = 0
