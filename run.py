import json
import os
import numpy as np
import pandas as pd
import genn_models

from h5py import File
#from sonata.circuit import File
from sonata.config import SonataConfig
from pygenn.genn_model import GeNNModel

from collections import defaultdict, namedtuple
from os import chdir
from time import perf_counter

def get_glif3_param_val_vars(components_dir, dynamics_params):
    with open(os.path.join(components_dir, dynamics_params)) as f:
        dynamics_params = json.load(f)
    print(dynamics_params)
    return {}, {}

node_id_lookup_dtype = np.dtype([("id", np.uint32), ("index", np.uint32)])

# Open top-level Sonata configu
chdir("v1_point")
cfg = SonataConfig.from_json("config.json")


# Open HDF5 and CSV files, specified by config
node_files = [(File(n["nodes_file"], "r"), 
               pd.read_csv(n["node_types_file"], index_col="node_type_id", 
                           usecols=lambda n: n in ["node_type_id", "dynamics_params", "pop_name"],
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
        # Build dataframe from required node data
        pop_df = pd.DataFrame(data={"node_type_id": pop["node_type_id"]},
                              index=pop["node_id"])
        pop_df = pop_df.join(node_types, on="node_type_id")

        # Group by population name and dynamic params if population has 
        # dynamics params i.e. if nodes aren't virtual, otherwise just by population name
        group_cols = (["pop_name", "dynamics_params"] if "dynamics_params" in node_types
                      else ["pop_name"])
    
        # Add list of tuples containing node ids and grouping terms to dictionary
        pop_node_dict[name] = [(df.index.to_numpy(), g)
                               for g, df in pop_df.groupby(group_cols)]
        print(f"\t\t{len(pop_node_dict[name])} homogeneous populations")

        # Create empty array to map node IDs to populations and indices within them
        start_time = perf_counter()
        node_id_lookup[name] = np.empty(len(pop_df), dtype=node_id_lookup_dtype)

        # Loop through newly-identified homogeneous populations and build lookup table
        for i, (nodes, g) in enumerate(pop_node_dict[name]):
            node_id_lookup[name]["id"][nodes] = i
            node_id_lookup[name]["index"][nodes] = nodes - nodes[0]
"""
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
                                    "source_pop_index": source_nodes["index"],
                                    "source_pop_id": source_nodes["id"],
                                    "target_pop_index": target_nodes["index"],
                                    "target_pop_id": target_nodes["id"]})

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
            
            # Group by delay and; source and target population id
            pop_edge_dict[name] = [grp + (df["source_pop_index"].to_numpy(), df["target_pop_index"].to_numpy())
                                   for grp, df in pop_group_df.groupby(["delay", "dynamics_params", 
                                                                        "source_pop_id", "target_pop_id"])]

            print(f"\t\t\t{len(pop_group_df)} edges in {len(pop_edge_dict[name])} homogeneous populations")
        print(f"\t\tBuilt edge dictionaries in {perf_counter() - start_time} seconds")
"""

# Create model and set timestamp
model = GeNNModel()
model.dT = cfg.dt

# Loop through populations
for pop_name, pops in pop_node_dict.items():
    # Loop through homogeneous GeNN populations within this
    for pop_nodes, pop_grouping in pops:
        assert len(pop_grouping) > 0
        print(f"{pop_name}_{pop_grouping[0]}")
        
        # If population has dynamics
        if len(pop_grouping) == 2:
            print("\tPoint process")
            param_vals, var_vals = get_glif3_param_val_vars(
                cfg.point_neuron_models_dir, pop_grouping[1])
        else:
            print("\tVirtual")
