import argparse
import os
from metrics.metric_utils import (
    get_numpy_matrix_from_adjacency,
    get_adj_graph_from_random_walks,
    get_total_nodes_and_edges_from_temporal_adj_list_in_time_range,
    get_adj_origina_graph_from_original_temporal_graph,
)
from metrics.metric_utils import (
    sample_adj_graph_multinomial_k_inductive,
    sample_adj_graph_topk,
)
from tgg_utils import *
import pandas as pd
import pickle
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm


### configurations
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path",
    help="full path of original dataset in csv format(start,end,time)",
    type=str,
)
parser.add_argument(
    "--config_path",
    help="full path of the folder where models and related data are saved during training",
    type=str,
)
parser.add_argument(
    "--num_of_sampled_graphs", help="No. of times , a graph was sampled", type=int
)
parser.add_argument(
    "--time_window",
    help="Size of each time window where data needs to be generated",
    type=int,
)
parser.add_argument("--topk_edge_sampling", help="Pick top K or sample Top K", type=int)
parser.add_argument("--l_w", default=10, help="lw", type=int)
parser.add_argument(
    "--model_name", help="name of the model need to be loaded", type=str
)
parser.add_argument(
    "--directed", action="store_true", help="Use directed graph"
)
parser.add_argument(
    "--savedir", default="results_best"
    , help="Directory to save generated graphs"
)


args = parser.parse_args()

data_path = args.data_path
config_dir = args.config_path
num_of_sampled_graphs = args.num_of_sampled_graphs
time_window = args.time_window
topk_edge_sampling = args.topk_edge_sampling
l_w = args.l_w
model_name = args.model_name
savedir = args.savedir
gg_savedir = str(Path(config_dir) / savedir)
Path(gg_savedir).mkdir(parents=True, exist_ok=True)
print("Running Graph Generation from Sampled Random Walks")
print(args)

### configurations

strictly_increasing_walks = True
num_next_edges_to_be_stored = 100
undirected = not args.directed

data = pd.read_csv(data_path)
if "u" in data.columns:
    data = data.rename(columns={"u": "start", "i": "end", "ts": "days"})

data = data[["start", "end", "days", "label"]]
graph_label_to_id = pickle.load(open(config_dir + "/graph_label_to_id.pkl", "rb"))
data["label"] = data["label"].map(graph_label_to_id)

print("Number of unique graph labels", len(data["label"].unique()))

node_set = set(data["start"]).union(set(data["end"]))
# print("number of nodes,",len(node_set))
node_set.update("end_node")
min_day = 1
max_day = max(data["days"])
min_day, max_day = int(min_day), int(max_day)

# print("Minimum, maximum timestamps",min(data['days']),max_days)
data = data.sort_values(by="days", inplace=False)
# print("number of interactions," ,data.shape[0])
# print(data.head())


vocab = pickle.load(open(config_dir + "/vocab.pkl", "rb"))
reverse_vocab = {value: key for key, value in vocab.items()}
end_node_id = vocab["end_node"]
# number of edges each day
day_indices = data['days'] // time_window
for label in set(data["label"]):
    for day_idx in set(day_indices):
        print(f"Label {label}, day_idx {day_idx}, number of edges {data[(data['label'] == label) & (day_indices == day_idx)].shape[0]}")

temporal_graph_original = defaultdict(
    lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
)
for start, end, day, label in tqdm(data[["start", "end", "days", "label"]].values):
    temporal_graph_original[label][day][start][end] += 1
    if undirected:
        temporal_graph_original[label][day][end][start] += 1

target_node_counts = defaultdict(dict)
target_edge_counts = defaultdict(dict)
time_labels = defaultdict(dict)
for label in tqdm(temporal_graph_original.keys()):
    for start_time in range(min_day, max_day, time_window):
        snapshot_idx = start_time // time_window
        
        tp, node_count = get_total_nodes_and_edges_from_temporal_adj_list_in_time_range(
            temporal_graph_original[label], start_time, start_time + time_window - 1
        )
        if undirected:
            tp = int(tp / 2)
        target_edge_counts[label][snapshot_idx] = tp
        target_node_counts[label][snapshot_idx] = node_count
        time_labels[label][snapshot_idx] = start_time

original_graphs = defaultdict(dict)
for label in temporal_graph_original.keys():
    for start_time in range(min_day, max_day, time_window):
        snapshot_idx = start_time // time_window
        
        og_snapshot = get_adj_origina_graph_from_original_temporal_graph(
            temporal_graph_original[label], start_time, start_time + time_window - 1
        )
        original_graphs[label][snapshot_idx] = og_snapshot

degree_distributions = defaultdict(defaultdict)
for label in original_graphs.keys():
    for snapshot_idx, graph in original_graphs[label].items():
        temp, _, _ = get_numpy_matrix_from_adjacency(graph)
        degree_distributions[label][snapshot_idx] = list(temp.sum(axis=0))

#            
pickle.dump(
    original_graphs,
    open(gg_savedir + f"/original_graphs.pkl", "wb"),
)
pickle.dump(
    time_labels, open(gg_savedir + f"/time_labels.pkl", "wb")
)
pickle.dump(max_day, open(gg_savedir + f"/max_days.pkl", "wb"))


def sequences_from_temporal_walks(generated_events, generated_times):
    sampled_walks = []
    lengths = []
    for i in range(generated_times.shape[0]):
        sample_walk_event = []
        sample_walk_time = []
        done = False
        j = 0
        while not done and j <= l_w:
            event = generated_events[i][j]
            time = generated_times[i][j]
            j += 1
            if event == end_node_id or time > max_day:
                done = True
            else:
                sample_walk_event.append(reverse_vocab[event])
                sample_walk_time.append(time)
        lengths.append(len(sample_walk_event))
        sampled_walks.append((sample_walk_event, sample_walk_time))
    print(
        "Mean length {} and Std deviation {}".format(
            str(np.mean(lengths)), str(np.std(lengths))
        )
    )
    sampled_walks = [item for item in sampled_walks if len(item[0]) >= 3]
    print(len(sampled_walks))
    return sampled_walks


sampled_graphs = defaultdict(dict)
for i in range(0, num_of_sampled_graphs):
    print(f"Loading generated events and times for sampled graph {i}")
    with open(config_dir + f"/results_{model_name}/generated_events_{i}.pkl", "rb") as f:
        generated_events = pickle.load(f)
    with open(config_dir + f"/results_{model_name}/generated_times_{i}.pkl", "rb") as f:
        generated_times = pickle.load(f)
    
    for lb_idx, label in enumerate(sorted(generated_events.keys())):

        print(f"Label {label}, generated events shape", generated_events[label].shape)
        sampled_walks = sequences_from_temporal_walks(generated_events[label], generated_times[label])
        adj_matrix_temporal_sampled = get_adj_graph_from_random_walks(sampled_walks, min_day, max_day, True)
        
        for start_time in range(min_day, max_day, time_window):
            snapshot_idx = start_time // time_window
            
            print("Snapshot idx: {}, start_time: {}".format(snapshot_idx, start_time))
             
            if target_edge_counts[label][snapshot_idx] == 0:
                sampled_lb_graph = {}
            elif topk_edge_sampling:
                sampled_lb_graph = sample_adj_graph_topk(
                        adj_matrix_temporal_sampled,
                        start_time,
                        start_time + time_window - 1,
                        target_edge_counts[label][snapshot_idx],
                        target_node_counts[label][snapshot_idx],
                        degree_distributions[label][snapshot_idx],
                        True,
                    )
            else:
                sampled_lb_graph = sample_adj_graph_multinomial_k_inductive(
                        adj_matrix_temporal_sampled,
                        start_time,
                        start_time + time_window - 1,
                        target_edge_counts[label][snapshot_idx],
                        target_node_counts[label][snapshot_idx],
                        degree_distributions[label][snapshot_idx],
                        True,
                    )
            sampled_graphs[label][snapshot_idx] = sampled_lb_graph
        #
        fp = open(gg_savedir + f"/sampled_graph_{i}.pkl", "wb")
        pickle.dump(sampled_graphs, fp)
        fp.close()
        print("Dumped the generated graph\n")


# sys.exit(0)
