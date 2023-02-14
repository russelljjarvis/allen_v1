from collections import defaultdict
from os import chdir
from time import perf_counter

from sonata.circuit import File
from sonata.config import SonataConfig

import numpy as np
import pandas as pd

import pygenn

# Open top-level Sonata configu
chdir("v1_point")
cfg = SonataConfig.from_json("config.json")

# Create model and set timestamp
model = pygenn.genn_model.GeNNModel()
model.dT = cfg.dt

# Open files, specified by config
# **NOTE** IDs overlap
node_files = [File(data_files=n["nodes_file"], 
                   data_type_files=n["node_types_file"])
              for n in cfg.networks["nodes"]]

edge_files = [File(data_files=n["edges_file"],
                   data_type_files=n["edge_types_file"],
                   require_magic=False)
              for n in cfg.networks["edges"]]

node_to_pop_id_dtype = np.dtype([("location", object), ("dynamics_params", object), 
                                 ("pop_id", np.uint32)])

edge_type_dtype = np.dtype([("delay", float), ("dynamics_params", object)])

# Loop through node files
print("Nodes")
pop_node_dict = {}
node_to_pop_id = {}
for f in node_files:
    # Loop through populations in each one
    # **NOTE** these aren't populations in the GeNN/PyNN sense
    for pop in f.nodes.populations:
        print(f"\t{pop.name}")

        # If this population has layer annotations
        node_df = pop.to_dataframe()
        if "layer" in node_df:
            start_time = perf_counter()

            # Group nodes by layer and dynamic parameter types and build dictionary
            # of (layer, dynamic params) tuples to numpy array of node IDs
            pop_node_dict[pop.name] = [(l, d, df["node_id"].to_numpy())
                                       for (l, d), df in node_df.groupby(["location", "dynamics_params"])]
            print(f"\t\t{len(pop_node_dict[pop.name])} homogeneous populations")
            print(f"\t\tBuilt dictionary in {perf_counter() - start_time} seconds")

            # Create empty array to map node IDs to populations and IDs within them
            start_time = perf_counter()
            node_to_pop_id[pop.name] = np.empty(len(node_df), dtype=node_to_pop_id_dtype)

            # Loop through newly-identified homogeneous populations and 
            for l, d, nodes in pop_node_dict[pop.name]:
                node_to_pop_id[pop.name]["location"][nodes] = l
                node_to_pop_id[pop.name]["dynamics_params"][nodes] = d
                node_to_pop_id[pop.name]["pop_id"][nodes] = nodes - nodes[0]

            print(f"\t\tBuilt ID lookup in {perf_counter() - start_time} seconds")

# Loop through edge files
print("Edges")
edge_source_dict = {}
edge_target_dict = {}
for f in edge_files:
    # Loop through populations in each one
    # **NOTE** these aren't populations in the GeNN/PyNN sense
    for pop in f.edges.populations:
        print(f"\t{pop.name} ({pop.source_population}->{pop.target_population})")

        # Get edge types as Pandas dataframe
        edge_types_df = pop.edge_types_table.to_dataframe()
        #print(edge_types_df

        # Check that they start at zero and are contiguous
        assert edge_types_df.index[0] == 0
        assert edge_types_df.index[-1] == (len(edge_types_df) - 1)


        edge_type_map = np.empty(len(edge_types_df), dtype=edge_type_dtype)
        for (d, p), df in edge_types_df.groupby(["delay", "dynamics_params"]):
            edge_type_map["delay"][df.index] = d
            edge_type_map["dynamics_params"][df.index] = p

        #assert False
        # **TODO** group edge type ids by dynamics_params i.e. receptor
        # Get node to pop ID lookup table for source and target populations
        source_node_to_pop_id = node_to_pop_id[pop.source_population]
        target_node_to_pop_id = node_to_pop_id[pop.target_population]

        # Loop through edges
        start_time = perf_counter()

        # Remap source and target node IDs into layer, dynamics and population ID
        # **YUCK** not sure how to do this within delving into internals
        source_pop_id = source_node_to_pop_id[:][pop._source_node_id_ds[()]]
        target_pop_id = target_node_to_pop_id[:][pop._target_node_id_ds[()]]

        edge_type = edge_type_map[:][pop._pop_group["edge_type_id"][()]]
        edge_df = pd.DataFrame(data={"edge_delay": edge_type["delay"],
                                     "edge_dynamics_params": edge_type["dynamics_params"],
                                     "source_location": source_pop_id["location"],
                                     "source_dynamics_params": source_pop_id["dynamics_params"],
                                     "source_pop_id": source_pop_id["pop_id"],
                                     "target_location": target_pop_id["location"],
                                     "target_dynamics_params": target_pop_id["dynamics_params"],
                                     "target_pop_id": target_pop_id["pop_id"]})
        edge_source_dict[pop.name] = [grp + (df["source_pop_id"].to_numpy(), df["target_pop_id"].to_numpy())
                                      for grp, df in edge_df.groupby(["edge_delay", "edge_dynamics_params", 
                                                                      "source_location", "source_dynamics_params",
                                                                      "target_location", "target_dynamics_params"])]

        print(f"\t\tlen(edge_df) edges in {len(edge_source_dict[pop.name])} homogeneous populations")
        print(f"\t\tBuilt edge dictionaries in {perf_counter() - start_time} seconds")
