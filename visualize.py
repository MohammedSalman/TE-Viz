import seaborn as sns
# import matplotlib
# matplotlib.use('agg')  # if not used we will get "Invalid DISPLAY variable" error when running on HPC.
import matplotlib.pyplot as plt
import numpy as np
from config import DRAW_AGGREGATED_TRAFFIC, Nu_of_TMs_Per_TOPO, DRAW_UTILIZATION, DRAW_minK_MaxK_DEVIATION
import copy
import time

ImageFileExtention = '.svg'


class Visualize:

    def __init__(self, df):
        self.df = df
        if DRAW_AGGREGATED_TRAFFIC:
            self.split_df_based_on_n_l_k_objtype_formtype_network_load()

        if DRAW_UTILIZATION:
            #  Each graph has to be drawn individually.
            self.iterate_through_df()
        if DRAW_minK_MaxK_DEVIATION:
            self.aggregate_based_on_n_l_k_obj_formtype_network_load(copy.deepcopy(self.df))

    def aggregate_based_on_n_l_k_obj_formtype_network_load(self, df):
        df = df.groupby(['n', 'k', 'l', 'formulation_type', 'obj_type', 'network_load'], as_index=False).agg(
            {'obj_val': 'mean'})
        self.split_df_based_on_objtype_formtype_network_load(df)

    def split_df_based_on_objtype_formtype_network_load(self, df):
        # only for rows with minK and maxK
        dataframes_dict = {}
        minK = min(df['k'].tolist())
        maxK = max(df['k'].tolist())
        for index, row in df.iterrows():
            key = str(row['obj_type'] + row['formulation_type'] + str(row['network_load']))
            if key in dataframes_dict:
                continue
            else:
                dataframes_dict[key] = df[(df['obj_type'] == row['obj_type']) &
                                          (df['formulation_type'] == row['formulation_type']) &
                                          (df['network_load'] == row['network_load']) &
                                          ((df['k'] == minK) | (df['k'] == maxK))]
        for key in dataframes_dict:
            text_display = key
            df = copy.deepcopy(dataframes_dict[key])
            current_obj = df.loc[df.groupby(['n', 'l', 'network_load'])['k'].transform('idxmin'), 'obj_val'].values
            optimal_obj = df['obj_val']
            df['obj_val'] = ((current_obj - optimal_obj) / current_obj) * 100  # the percentage gap
            df = df[df.k != df['k'].min()]
            df = df.pivot("l", "n", "obj_val")
            self.draw_deviation(df, text_display)

    def split_df_based_on_n_l_k_objtype_formtype_network_load(self):
        dataframes_dict = {}
        for index, row in self.df.iterrows():
            key = str(row['n']) + str(row['l']) + str(row['k']) + str(row['obj_type'] + str(row['formulation_type'])
                                                                      + str(row['network_load']))
            if key in dataframes_dict:
                continue
            else:
                dataframes_dict[key] = self.df[
                    (self.df['n'] == row['n']) & (self.df['l'] == row['l']) &
                    (self.df['k'] == row['k']) & (self.df['obj_type'] == row['obj_type'])
                    & (self.df['formulation_type'] == row['formulation_type']) & (
                            self.df['network_load'] == row['network_load'])]

        for key in dataframes_dict:
            self.DrawAggregatedTraficFlows(dataframes_dict[key])

    def draw_deviation(self, df, text_display):
        # textstr = self.text_to_display(df.to_dict('records')[0], figureType='deviation')

        # Draw a heatmap with the numeric values in each cell
        f, ax = plt.subplots()  # figsize=(9, 20)
        #  cmap = sns.diverging_palette(220, 10, as_cmap=True)
        #  cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.65, 0.95, text_display, transform=ax.transAxes, fontsize=8, bbox=props)
        cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
        # sns.heatmap(df, annot=True, annot_kws={"size": 8}, linewidths=.5, ax=ax)  # annot=True,
        sns.heatmap(df, annot=True, annot_kws={"size": 8}, linewidths=.5, ax=ax, cmap=cmap)  # annot=True,
        plt.show()
        filename = time.strftime("%Y%m%d-%H%M%S") + "_deviation_" + ImageFileExtention
        f.savefig(filename, dpi=600)

    @staticmethod
    def text_to_display(row, figureType=None):
        num_nodes = row['n']
        num_links = row['l']
        num_cand_paths = row['k']
        obj_type = row['obj_type']
        obj_val = row['obj_val']
        routing_strategy = row['formulation_type']
        network_load = row['network_load']
        if figureType == 'utilization':
            textstr = '\n'.join((
                'Number of nodes = %d' % num_nodes,
                'Number of links = %d' % num_links,
                'k = %d' % num_cand_paths,
                'Objective = %s' % obj_type,
                'Objective value = %.3f' % float(obj_val),
                'Routing strategy = %s' % routing_strategy,
                'Network load = %s' % network_load

            ))
        elif figureType == 'deviation':
            textstr = '\n'.join((
                'Objective = %s' % obj_type,
                'Routing strategy = %s' % routing_strategy,
                'Network load = %s' % network_load
            ))
        else:
            textstr = '\n'.join((
                'Number of nodes = %d' % num_nodes,
                'Number of links = %d' % num_links,
                'k = %d' % num_cand_paths,
                'Objective = %s' % obj_type,
                'Routing strategy = %s' % routing_strategy,
                'Network load = %s' % network_load,
                'Number of samples = %s' % Nu_of_TMs_Per_TOPO
            ))
        return textstr

    def iterate_through_df(self):
        for index, row in self.df.iterrows():
            textstr = self.text_to_display(row, figureType='utilization')

            textstr += '\n' + "Total available cap(%) = " + str(
                float('%.2f' % (row['percentage_available_cap']))) + '%'

            self.draw_utilization(row['links_utilization_and_residual'], textstr)

    def draw_utilization(self, utilization_dict, textstr):
        capacities = [utilization_dict[i]['capacity'] for i in utilization_dict]
        loads = [utilization_dict[i]['load'] for i in utilization_dict]

        N = len(capacities)
        theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
        # width = np.pi / 4 * np.random.rand(N)
        width = 0.02

        ax = plt.subplot(111, projection='polar', frameon=False)
        ax.set_xticklabels([])

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=8, bbox=props)

        bars = ax.bar(theta, capacities, width=width, bottom=0.0, color="skyblue")
        bars2 = ax.bar(theta, loads, width=width, bottom=0.0, color="dodgerblue")
        # for bar in bars:
        #     bar.set_alpha(0.50)
        # Use custom colors and opacity
        '''for r, bar in zip(radii, bars):
            bar.set_facecolor(plt.cm.viridis(r / 10.))
            bar.set_alpha(0.5)'''
        # filename = time.strftime("%Y%m%d-%H%M%S") + '_load' + ImageFileExtention
        # plt.savefig('png/' + filename, dpi=300)
        plt.show()

    def DrawAggregatedTraficFlows(self, df):
        textstr = self.text_to_display(df.to_dict('records')[0])
        k = df['k'].tolist()[0]
        aggregated_tf = df['aggregated_flow_value_per_path_index'].tolist()
        aggregated_tf = np.asarray(aggregated_tf)

        mins = aggregated_tf.min(0)
        maxes = aggregated_tf.max(0)
        means = aggregated_tf.mean(0)
        std = aggregated_tf.std(0)

        fig, ax = plt.subplots()

        # # create stacked errorbars:
        plt.errorbar(np.arange(len(aggregated_tf[0])), means, std, fmt='ok', lw=3)
        plt.errorbar(np.arange(len(aggregated_tf[0])), means, [means - mins, maxes - means], fmt='.k', ecolor='gray',
                     lw=1)
        plt.xlim(-0.25, k + 1)
        plt.xlabel("Shortest to Longest Path")
        plt.ylabel("Aggregated Traffic Per Path")

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.50, 0.70, textstr, transform=ax.transAxes, fontsize=10, bbox=props)

        plt.show()
        # filename = time.strftime("%Y%m%d-%H%M%S") + "_AggregatedTrafficDistribution_" + ImageFileExtention
        # plt.savefig(filename)
