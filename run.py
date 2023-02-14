from collections import defaultdict, namedtuple
from os import chdir
from time import perf_counter

from h5py import File
#from sonata.circuit import File
from sonata.config import SonataConfig


import numpy as np
import pandas as pd

import pygenn


node_to_pop_id_dtype = np.dtype([("location", object), ("dynamics_params", object), 
                                 ("pop_index", np.uint32)])

edge_type_dtype = np.dtype([("delay", float), ("dynamics_params", object)])

# Open top-level Sonata configu
chdir("v1_point")
cfg = SonataConfig.from_json("config.json")

# Create model and set timestamp
model = pygenn.genn_model.GeNNModel()
model.dT = cfg.dt

# Open files, specified by config
# **NOTE** IDs overlap
# **TODO** reduce width of datatypes
node_files = [(File(n["nodes_file"], "r"), 
               pd.read_csv(n["node_types_file"], index_col="node_type_id", 
                           usecols=lambda n: n in ["node_type_id", "dynamics_params", "location"],
                           delimiter=" ", skipinitialspace=True))
              for n in cfg.networks["nodes"]]

edge_files = [(File(n["edges_file"], "r"),
               pd.read_csv(n["edge_types_file"], index_col="edge_type_id", 
                           usecols=lambda n: n in ["edge_type_id", "delay", "dynamics_params"],
                           delimiter=" ", skipinitialspace=True))
              for n in cfg.networks["edges"]]

# Loop through node files
print("Nodes")
pop_node_dict = {}
node_id_lookup = {}
for nodes, node_types in node_files:
    # Loop through populations in each one
    # **NOTE** these aren't populations in the GeNN/PyNN sense
    for name, pop in nodes["nodes"].items():
        print(f"\t{name}")
        
        # If node has dynamics ie is 'real' neuron
        # **TODO** something model_type based
        if "dynamics_params" in node_types:
            # Build dataframe from required node data
            pop_df = pd.DataFrame(data={"node_type_id": pop["node_type_id"]},
                                  index=pop["node_id"])
            pop_df = pop_df.join(node_types, on="node_type_id")
            
            
            # Group nodes by location and dynamic parameter types and build dictionary
            # of (location, dynamic params) tuples to numpy array of node IDs
            start_time = perf_counter()
            pop_node_dict[name] = [(l, d, df.index.to_numpy())
                                   for (l, d), df in pop_df.groupby(["location", "dynamics_params"])]
            print(f"\t\t{len(pop_node_dict[name])} homogeneous populations")
            print(f"\t\tBuilt dictionary in {perf_counter() - start_time} seconds")

            # Create empty array to map node IDs to populations and IDs within them
            start_time = perf_counter()
            node_id_lookup[name] = np.empty(len(pop_df), dtype=node_to_pop_id_dtype)

            # Loop through newly-identified homogeneous populations and 
            # **TODO** abstract away unique variables into index so this works with virtual nodes
            for l, d, nodes in pop_node_dict[name]:
                node_id_lookup[name]["location"][nodes] = l
                node_id_lookup[name]["dynamics_params"][nodes] = d
                node_id_lookup[name]["pop_index"][nodes] = nodes - nodes[0]

            print(f"\t\tBuilt ID lookup in {perf_counter() - start_time} seconds")

# Loop through edge files
print("Edges")
pop_edge_dict = {}
for edges, edge_types in edge_files:
    # Loop through populations in each one
    # **NOTE** these aren't populations in the GeNN/PyNN sense
    for name, pop in edges["edges"].items():
        source_node_pop = pop["source_node_id"].attrs["node_population"]
        target_node_pop = pop["target_node_id"].attrs["node_population"]
        print(f"\t{name} ({source_node_pop}->{target_node_pop})")
        start_time = perf_counter()
        
        # Lookup source and target nodes
        source_nodes = node_id_lookup[source_node_pop][:][pop["source_node_id"][()]]
        target_nodes = node_id_lookup[target_node_pop][:][pop["target_node_id"][()]]
        
        # Build dataframe from required edge data
        pop_df = pd.DataFrame(data={"edge_group_id": pop["edge_group_id"],
                                    "edge_group_index": pop["edge_group_index"],
                                    "edge_type_id": pop["edge_type_id"],
                                    "source_pop_index": source_nodes["pop_index"],
                                    "source_location": source_nodes["location"],
                                    "source_dynamics_params": source_nodes["dynamics_params"],
                                    "target_pop_index": target_nodes["pop_index"],
                                    "target_location": target_nodes["location"],
                                    "target_dynamics_params": target_nodes["dynamics_params"]})

        # Group by group ID
        for group_id, pop_group_df in pop_df.groupby("edge_group_id"):
            print(f"\t\tGroup {group_id}")

            # Build dataframe from required group data
            group = pop[str(group_id)]
            group_df = pd.DataFrame(data={"syn_weight": group["syn_weight"]})

            # Join with pop_df
            # **THINK** is this right sort of join
            pop_group_df = pop_group_df.join(group_df, on="edge_group_index")
            
            # Join with edge types
            pop_group_df = pop_group_df.join(edge_types, on="edge_type_id")
            
            pop_edge_dict[name] = [grp + (df["source_pop_index"].to_numpy(), df["target_pop_index"].to_numpy())
                                   for grp, df in pop_group_df.groupby(["delay", "dynamics_params", 
                                                                        "source_location", "source_dynamics_params",
                                                                        "target_location", "target_dynamics_params"])]

            print(f"\t\t\tlen(pop_group_df) edges in {len(pop_edge_dict[name])} homogeneous populations")
        print(f"\t\tBuilt edge dictionaries in {perf_counter() - start_time} seconds")
