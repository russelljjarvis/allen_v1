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
node_read_start_time = perf_counter()
pop_node_dict = {}
node_id_lookup = {}
pop_id_lookup = defaultdict(list)
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

        # Loop through newly-identified homogeneous populations
        for i, (indices, g) in enumerate(pop_node_dict[name]):
            # Build lookup from node indices to homogeneous population id and indices within that
            node_id_lookup[name]["id"][indices] = i
            node_id_lookup[name]["index"][indices] = np.arange(len(indices))

            # Build second lookup from population indices to node indices
            pop_id_lookup[name].append(indices)

input_read_start_time = perf_counter()
print(f"\t\t{input_read_start_time - node_read_start_time} seconds")

# Loop through inputs provided by config
print("\tInputs")
input_dict = {}
for name, input in cfg.inputs.items():
    # Check input is of supported type
    assert input["input_type"] == "spikes"
    assert input["module"] == "h5"

    # 'node_set' CAN be used for something else but, 
    # here, it appears to be used for specifying population
    assert input["node_set"] in pop_node_dict

    # Open spike file
    input_dict[input["node_set"]] = File(input["input_file"], "r")

edge_read_start_time = perf_counter()
print(f"\t\t{edge_read_start_time - input_read_start_time} seconds")

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

neuron_create_start_time = perf_counter()
print(f"\t\t{neuron_create_start_time - edge_read_start_time} seconds")

# Create model and set timestamp
model = GeNNModel("float", "v1_point", generateSimpleErrorHandling=True)
model.dT = cfg.dt
model._model.set_merge_postsynaptic_models(True)
model._model.set_default_narrow_sparse_ind_enabled(True)

# Loop through node populations
print("Creating GeNN model")
print("\tCreating GeNN neuron populations")
genn_neuron_pop_dict = defaultdict(list)
for pop_name, pops in pop_node_dict.items():
    if pop_name in input_dict:
        # Lookup source and target nodes
        # **NOTE** SOME spike files seem to have an additional level of indirection here
        input_spikes = input_dict[pop_name]["spikes"]

        # Use node lookup to get population ids and indices corresponding to nodes
        # **NOTE** in SOME spike files, this field is called "node_ids"
        input_nodes = node_id_lookup[pop_name][:][input_spikes["gids"][()]]

        # Load spike data
        input_spikes_df = pd.DataFrame(data={"timestamps": input_spikes["timestamps"],
                                             "pop_index": input_nodes["index"],
                                             "pop_id": input_nodes["id"]})
        # Build dictionary with input spikes grouped by population ID
        pop_input_spikes = {id: (df["timestamps"].to_numpy(), df["pop_index"].to_numpy())
                            for id, df in input_spikes_df.groupby("pop_id")}

    # Loop through homogeneous GeNN populations within this
    for pop_id, (pop_nodes, pop_grouping) in enumerate(pops):
        assert len(pop_grouping) > 0
        num_neurons = len(pop_nodes)
        genn_pop_name = f"{pop_name}_{pop_grouping[0]}_{pop_id}"

        # If population has dynamics
        if len(pop_grouping) == 2:
            # Convert dynamics_params to GeNN parameter and variable values
            param_vals, var_vals = get_glif3_param_val_vars(cfg, pop_grouping[1])

            # Add population
            genn_pop = model.add_neuron_population(
                genn_pop_name, num_neurons, genn_models.glif3,
                param_vals, var_vals)
        # Otherwise
        else:
            # Check that input spikes were read for this population
            assert pop_id in pop_input_spikes

            # Calculate number of spikes per-neuron and then cumulative index
            end_spikes = np.cumsum(np.bincount(pop_input_spikes[pop_id][1], 
                                   minlength=num_neurons))
            assert len(end_spikes) == num_neurons

            # Build start spikes from end spikes
            start_spikes = np.empty_like(end_spikes)
            start_spikes[0] = 0
            start_spikes[1:] = end_spikes[:-1]

            # Sort events first by neuron id and then by time and use to order spike times
            spike_times = pop_input_spikes[pop_id][0][np.lexsort(pop_input_spikes[pop_id])]

            # Build spike source array
            genn_pop = model.add_neuron_population(
                genn_pop_name, num_neurons, "SpikeSourceArray",
                {}, {"startSpike": start_spikes, "endSpike": end_spikes})
            genn_pop.set_extra_global_param("spikeTimes",  spike_times)
            genn_pop.spike_recording_enabled = True

        # Add to dictionary
        # **NOTE** indexing will be the same as pop_node_dict
        genn_neuron_pop_dict[pop_name].append(genn_pop)

synapse_create_start_time = perf_counter()
print(f"\t\t{synapse_create_start_time - neuron_create_start_time} seconds")

# Loop through edge populations
print("\tCreating GeNN synapse populations")
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
        tau_syn = get_glif3_tau_syn(cfg, target_dynamics_params)

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
            genn_models.psc_alpha, {"tau": tau_syn[receptor_index - 1]}, {"x": 0.0})

        # Set sparse connectivity
        pop.set_sparse_connections(source_pop_index, target_pop_index)

build_start_time = perf_counter()
print(f"\t\t{build_start_time - synapse_create_start_time} seconds")

# Build model
print("Building GeNN model")
mem_usage = model.build()
print(f"\tModel requires {mem_usage.get_device_mbytes()}MB device memory and {mem_usage.get_host_mbytes()}MB host memory")

load_start_time = perf_counter()
print(f"\t{load_start_time - build_start_time} seconds")

# Load model
print("Loading GeNN model")
duration_ms = cfg.run["duration"]
model.load(num_recording_timesteps=int(round(duration_ms / cfg.dt)))

run_start_time = perf_counter()
print(f"\t{run_start_time - load_start_time} seconds")

# Simulate model
print(f"Simulating GeNN model for {duration_ms}ms")
while model.t < duration_ms:
     model.step_time()

model.pull_recording_buffers_from_device()

run_stop_time = perf_counter()
print(f"\t{run_stop_time - run_start_time} seconds")

# Loop through population
output_spike_timestamps = []
output_spike_node_ids = []
output_spike_pop_names = []
for pop_name, genn_pops in genn_neuron_pop_dict.items():
    # Loop through homogeneous GeNN populations
    for pop_id, genn_pop in enumerate(genn_pops):
        # If spike recording is enabled
        if genn_pop.spike_recording_enabled:
            # Extract spike recording data from population
            st, sid = genn_pop.spike_recording_data

            # Add numpy arrays to lists
            output_spike_timestamps.append(st)
            output_spike_node_ids.append(pop_id_lookup[pop_name][pop_id][sid])
            output_spike_pop_names.extend([pop_name] * len(st))

# Assemble dataframe and write to CSV
output_spike_df = pd.DataFrame(data={"timestamps": np.concatenate(output_spike_timestamps),
                                     "population": output_spike_pop_names,
                                     "node_ids": np.concatenate(output_spike_node_ids)})
output_spike_df.to_csv("spikes.csv", sep=" ")
