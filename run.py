import json
import os
import numpy as np
import pandas as pd
import genn_models

from h5py import File
from sonata.config import SonataConfig
from pygenn.genn_model import GeNNModel

from collections import defaultdict, namedtuple
from os import chdir
from time import perf_counter

# Get GeNN neuron parameter and variable values from a GLIF3 model dynamics params file
def get_glif3_param_val_vars(cfg, dynamics_params):
    with open(os.path.join(cfg.point_neuron_models_dir, dynamics_params)) as f:
        dynamics_params = json.load(f)

    assert len(dynamics_params["asc_init"]) == 2
    assert len(dynamics_params["asc_amps"]) == 2
    assert len(dynamics_params["asc_decay"]) == 2

    asc_decay = np.asarray(dynamics_params["asc_decay"])
    r = np.asarray([1.0, 1.0])  # NEST default
    asc_decay_rates = np.exp(-asc_decay * cfg.dt)
    asc_stable_coeff = (1.0 / asc_decay / cfg.dt) * (1.0 - asc_decay_rates)
    asc_refractory_decay_rates = r * np.exp(-asc_decay * dynamics_params["t_ref"])

    param_vals = {
        "C": dynamics_params["C_m"] / 1000,  # pF -> nF
        "G": dynamics_params["g"] / 1000,  # nS -> uS
        "El": dynamics_params["E_L"],
        "spike_cut_length": round(dynamics_params["t_ref"] / cfg.dt),
        "th_inf": dynamics_params["V_th"],
        "V_reset": dynamics_params["V_reset"],  # BMTK rounds to 3rd decimal
        "asc_amp_array_1": dynamics_params["asc_amps"][0] / 1000,  # pA->nA
        "asc_amp_array_2": dynamics_params["asc_amps"][1] / 1000,  # pA->nA
        "asc_stable_coeff_1": asc_stable_coeff[0],
        "asc_stable_coeff_2": asc_stable_coeff[1],
        "asc_decay_rates_1": asc_decay_rates[0],
        "asc_decay_rates_2": asc_decay_rates[1],
        "asc_refractory_decay_rates_1": asc_refractory_decay_rates[0],
        "asc_refractory_decay_rates_2": asc_refractory_decay_rates[1]}

    var_vals = {
        "V": dynamics_params["V_m"],
        "refractory_countdown": -1,
        "ASC_1": dynamics_params["asc_init"][0] / 1000,  # pA -> nA
        "ASC_2": dynamics_params["asc_init"][1] / 1000}  # pA -> nA

    return param_vals, var_vals

# Get tau_syn array from a GLIF3 model dynamics params file
def get_glif3_tau_syn(cfg, dynamics_params):
    with open(os.path.join(cfg.point_neuron_models_dir, dynamics_params)) as f:
        dynamics_params = json.load(f)

    return dynamics_params["tau_syn"]

# Get receptor index from a static synapse model dynamics params file
def get_static_synapse_receptor_index(cfg, dynamics_params):
    with open(os.path.join(cfg.synaptic_models_dir, dynamics_params)) as f:
        dynamics_params = json.load(f)

    return dynamics_params["receptor_type"]

node_id_lookup_dtype = np.dtype([("id", np.uint32), ("index", np.uint32)])

# Open top-level Sonata configu
chdir("v1_point")
cfg = SonataConfig.from_json("config.json")


# Open HDF5 and CSV files, specified by config
# **NOTE** because we join these later we only read columns we use
# **NOTE** lambda lets you read csvs with optional columns in usecols
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
print("Reading SONATA model description")
print("\tNodes")
pop_node_dict = {}
node_id_lookup = {}
for nodes, node_types in node_files:
    # Loop through populations in each one
    # **NOTE** these aren't populations in the GeNN/PyNN sense
    for name, pop in nodes["nodes"].items():
        print(f"\t\t{name}")
        # Build dataframe from required node data
        # **TODO** check that all model_template are nest:glif_psc
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
        print(f"\t\t\t{len(pop_df)} neurons in {len(pop_node_dict[name])} homogeneous populations")

        # Create empty array to map node IDs to populations and indices within them
        start_time = perf_counter()
        node_id_lookup[name] = np.empty(len(pop_df), dtype=node_id_lookup_dtype)

        # Loop through newly-identified homogeneous populations and build lookup table
        for i, (indices, g) in enumerate(pop_node_dict[name]):
            node_id_lookup[name]["id"][indices] = i
            node_id_lookup[name]["index"][indices] = np.arange(len(indices))

# Loop through edge files
print("\tEdges")
pop_edge_dict = {}
for edges, edge_types in edge_files:
    # Loop through populations in each one
    # **NOTE** these aren't populations in the GeNN/PyNN sense
    for name, pop in edges["edges"].items():
        source_node_pop = pop["source_node_id"].attrs["node_population"]
        target_node_pop = pop["target_node_id"].attrs["node_population"]
        print(f"\t\t{name} ({source_node_pop}->{target_node_pop})")
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
            print(f"\t\t\tGroup {group_id}")

            # Build dataframe from required group data
            group = pop[str(group_id)]
            group_df = pd.DataFrame(data={"syn_weight": group["syn_weight"]})

            # Join with pop_df
            # **THINK** is this right sort of join
            pop_group_df = pop_group_df.join(group_df, on="edge_group_index")

            # Join with edge types
            pop_group_df = pop_group_df.join(edge_types, on="edge_type_id")

            # Group by delay and; source and target population id
            hom_pops = [grp + (df["source_pop_index"].to_numpy(), df["target_pop_index"].to_numpy(), df["syn_weight"].to_numpy())
                        for grp, df in pop_group_df.groupby(["delay", "dynamics_params", 
                                                             "source_pop_id", "target_pop_id"])]
            pop_edge_dict[(name, source_node_pop, target_node_pop)] = hom_pops 
            print(f"\t\t\t\t{len(pop_group_df)} edges in {len(hom_pops)} homogeneous populations")
        print(f"\t\t\tBuilt edge dictionaries in {perf_counter() - start_time} seconds")

# Create model and set timestamp
model = GeNNModel()
model.dT = cfg.dt
model.fuse_postsynaptic_models = True
model.default_narrow_sparse_ind_enabled = True

# Loop through node populations
print("Building GeNN model")
print("\tBuilding GeNN neuron populations")
genn_neuron_pop_dict = defaultdict(list)
for pop_name, pops in pop_node_dict.items():
    # Loop through homogeneous GeNN populations within this
    for pop_id, (pop_nodes, pop_grouping) in enumerate(pops):
        assert len(pop_grouping) > 0
        genn_pop_name = f"{pop_name}_{pop_grouping[0]}_{pop_id}"

        # If population has dynamics
        if len(pop_grouping) == 2:
            # Convert dynamics_params to GeNN parameter and variable values
            param_vals, var_vals = get_glif3_param_val_vars(cfg, pop_grouping[1])
            
            # Add population
            genn_pop = model.add_neuron_population(
                genn_pop_name, len(pop_nodes), genn_models.glif3,
                param_vals, var_vals)
        else:
            # **TEMP**
            genn_pop = model.add_neuron_population(
                genn_pop_name, len(pop_nodes), "SpikeSource",
                {}, {})
        
        # Add to dictionary
        # **NOTE** indexing will be the same as pop_node_dict
        genn_neuron_pop_dict[pop_name].append(genn_pop)

# Loop through edge populations
print("\tBuilding GeNN synapse populations")
genn_synapse_pop_dict = defaultdict(list)
for (pop_name, source_node_pop, target_node_pop), pops in pop_edge_dict.items():
    # Loop through homogeneous GeNN populations within this
    # **TODO** namedtuple
    for pop_id, (delay, dynamics_params, source_pop_id, target_pop_id,
                 source_pop_index, target_pop_index, syn_weight) in enumerate(pops):
        genn_pop_name = f"{pop_name}_{pop_id}"

        # Read receptor index from dynamics params
        receptor_index = get_static_synapse_receptor_index(cfg, dynamics_params)

        # Get dynamics params used by target population
        target_dynamics_params = pop_node_dict[target_node_pop][target_pop_id][1][1]

        # From these, read tau_syn for this synapse group
        tau_syn = get_glif3_tau_syn(cfg, target_dynamics_params)[receptor_index]

        # Round delay
        delay = int(round(delay / cfg.dt))
        
        # Convert weight from nS to uS
        syn_weight = syn_weight / 1000.0
        
        # Add population to model
        pop = model.add_synapse_population(
            genn_pop_name, "SPARSE_INDIVIDUALG", delay,
            genn_neuron_pop_dict[source_node_pop][source_pop_id],
            genn_neuron_pop_dict[target_node_pop][target_pop_id],
            "StaticPulse", {}, {"g": syn_weight}, {}, {},
            genn_models.psc_alpha, {"tau": tau_syn}, {"x": 0.0})

        # Set sparse connectivity
        pop.set_sparse_connections(source_pop_index, target_pop_index)

# Build model
mem_usage = model.build()
print(f"Model requires {mem_usage.get_device_mbytes()}MB device memory and {mem_usage.get_host_mbytes()}MB host memory")