import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pylab as pylab
import scipy.stats as stats
from h5py import File
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

def plot_raster_query(ax,spikes,nodes_group,node_types_df,cmap, plot_order, twindow=[0,3], marker=".", lw=0,s=10):
    '''
    Plot raster colored according to a query.
    Query's key defines node selection and the corresponding values defines color

    Parameters:
    -----------
        ax: matplotlib axes object
            axes to use
        spikes: tuple of numpy arrays
            includes [times, gids]
        nodes_group: h5py file
        cmap: dict
            key: query string, value:color
        twindow: tuple
            [start_time,end_time]
        plot_order: order to plot cell classes - list

    '''
    tstart = twindow[0]
    tend = twindow[1]

    ix_t = np.where((spikes["timestamps"]>tstart) & (spikes["timestamps"]<tend))[0]

    spike_times = spikes["timestamps"][ix_t]
    spike_gids = spikes["node_ids"][ix_t]

    v1_nodes = nodes_group["nodes"]["v1"]
    v1_nodes_df = pd.DataFrame(data={"node_group_id": v1_nodes['node_group_id'], 
                                     "node_group_index": v1_nodes['node_group_index'], 
                                     "node_type_id": v1_nodes['node_type_id']},
                               index=v1_nodes['node_id'])
    counter = 0
    # Grey boxes to be drawn to delineate layers with the below colormap
    patch_colors = ['grey', 'w', 'grey', 'w', 'grey']
    for query in plot_order:
        col = cmap[query]
        
        # Get node type IDs which are part of population starting with query
        query_node_types_df = node_types_df[node_types_df["pop_name"].str.startswith(query)]

        # Join with V1 nodes to get all nodes that are part of this population
        query_nodes_df = v1_nodes_df.join(query_node_types_df, on="node_type_id", how="inner")
  
        # Loop through all groups of nodes which this encompasses
        for group_id, query_group_nodes_df in query_nodes_df.groupby("node_group_id"):
            group = v1_nodes[str(group_id)]
            group_df = pd.DataFrame(data={"tuning_angle": group["tuning_angle"]})
            
            # Join with pop_df
            # **THINK** is this right sort of join
            pop_group_df = query_group_nodes_df.join(group_df, on="node_group_index")

            gids_query = pop_group_df.index
            tuning_angles = pop_group_df["tuning_angle"]

            print (query, "ncells:", len(gids_query), col)

            ix_g = np.in1d(spike_gids, gids_query)

            spikes_gids_temp = spike_gids.to_numpy()[ix_g]
            gids_temp = stats.rankdata(tuning_angles, method='dense') - 1
            gids_temp = gids_temp + counter
            for i, gid in enumerate(gids_query):
                inds = np.where(spikes_gids_temp == gid)[0]
                spikes_gids_temp[inds] = gids_temp[i]

            counter += len(gids_query)


            ax.plot(spike_times[ix_g], spikes_gids_temp,
                        marker= marker,
                        color = col,
                        label=query,
                        lw=lw,
                        markersize = s
                        );
        # Plotting for boxes to be drawn to delineate layers
        if ('Htr3a' in query):
            if 'xy' not in locals():
                xy = (0,0)
                w = twindow[1] + 500
                h = counter
                h_cumsum = h
            else:
                xy = (0, h_cumsum)
                h = counter - h_cumsum
                h_cumsum += h

            ax.add_patch(Rectangle(xy, w, h, color=patch_colors.pop(), alpha=0.2))






if __name__ == '__main__':

    # Spikes file to load and plot
    spikes_file_name = 'v1_point/spikes.csv'
    # spikes_file_name = 'biophysical/spikes_flash_trial0.txt'
    # spikes_file_name = 'biophysical/spikes_naturalMovie_trial0.txt'
    spikes = pd.read_csv(spikes_file_name, sep=" ", skipinitialspace=True)

    # Nodes file to read
    nodes_group = File('v1_point/network/v1_nodes.h5', 'r')
    node_types_df = pd.read_csv('v1_point/network/v1_node_types.csv', sep=" ", skipinitialspace=True, index_col="node_type_id")

    # Color map to be used
    cmap = {
        "i1Htr3a": 'indigo',
        "e23": 'firebrick',
        "i23Pvalb": 'blue',
        "i23Sst": 'forestgreen',
        "i23Htr3a": 'indigo',
        "e4": 'firebrick',
        "i4Pvalb": 'blue',
        "i4Sst": 'forestgreen',
        "i4Htr3a": 'indigo',
        "e5": 'firebrick',
        "i5Pvalb": 'blue',
        "i5Sst": 'forestgreen',
        "i5Htr3a": 'indigo',
        "e6": 'firebrick',
        "i6Pvalb": 'blue',
        "i6Sst": 'forestgreen',
        "i6Htr3a": 'indigo',
    }

    # To plot L6 at the bottom and L1 at the top
    plot_order = [
        "e6", "i6Pvalb", "i6Sst", "i6Htr3a",
        "e5", "i5Pvalb", "i5Sst", "i5Htr3a",
        "e4", "i4Pvalb", "i4Sst", "i4Htr3a",
        "e23", "i23Pvalb", "i23Sst", "i23Htr3a",
        "i1Htr3a",
        ]

    # Plot the results
    fig, ax = plt.subplots(figsize=(24, 16))
    plot_raster_query(ax, spikes, nodes_group, node_types_df, cmap, plot_order, twindow=[0, 3000], marker=".", lw=0, s=2.)
    ax.set_xlabel('Time (ms)');
    ax.set_ylabel('Neuron ID');

    """

    # For gratings or movie: 500ms grey then 2500ms stimulus
    if 'flash' not in spikes_file_name:
        plt.plot([500, 3000], [-1000, -1000], color='k', lw=5)
        plt.plot([0, 500], [-7000, -7000], color='k', lw=5)
        plt.plot([500, 500], [-1000, -7000], color='k', lw=5)
        ax.set_ylim([-10000, 54000])

    # For flashes: grey for 500ms then ON for 250ms then grey then OFF for 250ms then grey again
    else:
        plt.plot([0, 500], [-7000, -7000], color='k', lw=5)
        plt.plot([500, 750], [-1000, -1000], color='k', lw=5)
        plt.plot([750, 1750], [-7000, -7000], color='k', lw=5)
        plt.plot([1750, 2000], [-13000, -13000], color='k', lw=5)
        plt.plot([2000, 2500], [-7000, -7000], color='k', lw=5)

        plt.plot([500, 500], [-1000, -7000], color='k', lw=5)
        plt.plot([750, 750], [-1000, -7000], color='k', lw=5)
        plt.plot([1750, 1750], [-13000, -7000], color='k', lw=5)
        plt.plot([2000, 2000], [-13000, -7000], color='k', lw=5)
        ax.set_ylim([-16000, 54000])
    """
    plt.show()


