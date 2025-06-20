import argparse
import os
from metrics.metric_utils import (
    get_numpy_matrix_from_adjacency,
    get_adj_graph_from_random_walks,
    get_total_nodes_and_edges_from_temporal_adj_list_in_time_range,
    get_adj_origina_graph_from_original_temporal_graph,
    convert_graph_from_defauldict_to_dict
)
from metrics.metric_utils import (
    sample_adj_graph_multinomial_k_inductive,
    sample_adj_graph_topk,
)
from tgg_utils import *
import pandas as pd
import pickle
from collections import defaultdict
import sys
from pathlib import Path
import h5py
from tqdm import tqdm
import glob
from copy import deepcopy

### configurations
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path",
    help="full path of original dataset in csv format(start,end,time)",
    type=str,
)
parser.add_argument(
    "--sampled_data_dir",
    help="full path of sampled dataset in csv format(start,end,time)",
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
sampled_data_dir = args.sampled_data_dir
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

#
smapled_graphs = {}
for fname in glob.glob(sampled_data_dir + f"/sampled_graph_*_*.csv"):
    if not fname.endswith(".csv"):
        continue
    label, idx = fname.split("_")[-2], fname.split("_")[-1].split(".")[0]
    label, idx = float(label), int(idx)
    df = pd.read_csv(fname)
    if idx in smapled_graphs:
        smapled_graphs[idx].append(df)
    else:
        smapled_graphs[idx] = [df]

for idx, dfs in smapled_graphs.items():
    sampled_data = pd.concat(dfs)
    sampled_data.to_csv(sampled_data_dir + f"/sampled_graph_{idx}.csv", index=False)
    
# exit()
# exit(1)
#
data = pd.read_csv(data_path)
if "u" in data.columns:
    data = data.rename(columns={"u": "start", "i": "end", "ts": "days"})

data = data[["start", "end", "days", "label"]]

graph_label_to_id = pickle.load(open(config_dir + "/../graph_label_to_id.pkl", "rb"))
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


vocab = pickle.load(open(config_dir + "/../vocab.pkl", "rb"))
reverse_vocab = {value: key for key, value in vocab.items()}
end_node_id = vocab["end_node"]

os.makedirs(config_dir + f"/results_{model_name}", exist_ok=True)

def extract_snapshots(data, min_day, max_day, time_window, save_name='original_graphs'):
    
    temporal_graph = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    )
    for start, end, day, label in tqdm(data[["start", "end", "days", "label"]].values):
        temporal_graph[label][day][start][end] += 1
        if undirected:
            temporal_graph[label][day][end][start] += 1

    graphs = defaultdict(dict)
    for label in temporal_graph.keys():
        for start_time in range(min_day, max_day, time_window):
            snapshot_idx = start_time // time_window
            
            snapshot = get_adj_origina_graph_from_original_temporal_graph(
                temporal_graph[label], start_time, start_time + time_window - 1
            )
            old_snapshot = deepcopy(snapshot)
            snapshot = defaultdict(lambda: defaultdict(lambda: 0))
            for start,adj_list in old_snapshot.items():
                for end, ct in adj_list.items():
                    if ct >0:
                        snapshot[int(start)][int(end)] = 1
            # convert to dict 
            snapshot = convert_graph_from_defauldict_to_dict(snapshot)
            
            graphs[label][snapshot_idx] = snapshot

    savepath = gg_savedir + f"/{save_name}.pkl"
    pickle.dump(
        graphs, open(savepath, "wb"),
    )
    print(f"Saved {savepath}")
    
#

extract_snapshots(data, min_day, max_day, time_window, save_name='original_graphs')

for i in range(num_of_sampled_graphs):
    sampled_data = pd.read_csv(sampled_data_dir + f"/sampled_graph_{i}.csv")
    if "u" in sampled_data.columns:
        sampled_data = sampled_data.rename(columns={"u": "start", "i": "end", "ts": "days"})
        
    sampled_data = sampled_data[["start", "end", "days", "label"]]
    sampled_data["label"] = sampled_data["label"].map(graph_label_to_id)
    extract_snapshots(sampled_data, min_day, max_day, time_window, save_name=f'sampled_graph_{i}')

# sys.exit(0)
