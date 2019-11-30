import logging
from config import N, L, Nu_of_TOPOs_Per_N_L
from topology import Topology
from multiprocessing import Process
from programsettings import PARALLELISM, CPU_COUNT
from visualize import Visualize
import time
from itertools import product
import pandas as pd
import random
import os


def chunks(ll, nn):
    for index in range(0, len(ll), nn):
        yield ll[index:index + nn]


if __name__ == '__main__':

    # Saving style of the shared object

    OutputDirName = time.strftime("%Y-%m-%d-H%H-M%M-S%S") + '_PID-' + str(os.getpid())
    try:
        os.makedirs(OutputDirName)
    except OSError:
        print("Couldn't make the directory")
        exit(0)

    if isinstance(N, int):
        N = [N]
    if isinstance(L, int):
        L = [L]
    tic = time.time()
    df = pd.DataFrame()
    procs = []

    random.shuffle(N)
    random.shuffle(L)

    # for n, l in product(N, L):
    for _ in range(Nu_of_TOPOs_Per_N_L):
        for n, l in random.sample(list(product(N, L)), len(N) * len(L)):

            if PARALLELISM:
                proc = Process(target=Topology, args=(n, l, OutputDirName))
                procs.append(proc)
            else:
                Topology(n, l, OutputDirName)

    if PARALLELISM:
        for i in chunks(procs, CPU_COUNT):
            for j in i:
                j.start()
            for j in i:
                j.join()

    print()

    # print(type(list_of_dict))
    # print(list_of_dict[0])
    # for item in list_of_dict:
    #     df = pd.concat([df, pd.DataFrame([item])], ignore_index=True)
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(OutputDirName):
        for file in f:
            if '.pkl' in file:
                files.append(os.path.join(r, file))
    dataframes = []
    startMergingTime = time.time()
    for f in files:
        df = pd.read_pickle(f)
        dataframes.append(df)
        # print(df)
    output = pd.concat(dataframes, ignore_index=True)
    print("Output merged in: ", time.time() - startMergingTime, " seconds.")
    print("Program finished after: ", time.time() - tic, " seconds.")

    try:
        output.to_pickle("pickled_df")
    except PermissionError:
        filename = time.strftime("%Y%m%d-%H%M%S")
        output.to_pickle(filename)
        # print("pickled file saved in: " + filename)
        logging.warning('pickled file saved in: ' + filename)
        # write file in another name

    try:
        output.to_csv("pickled_df.csv")
    except PermissionError:
        filename = time.strftime("%Y%m%d-%H%M%S") + '.csv'
        # filename = "{0}{1}".format(time.strftime("%Y%m%d-%H%M%S"), '.csv')
        output.to_csv(filename, index=False)
        # print("csv file saved in: " + filename)
        logging.warning('csv file saved in: ' + filename)
    Visualize(output)
