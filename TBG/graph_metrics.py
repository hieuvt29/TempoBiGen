from metrics.metric_utils import get_numpy_matrix_from_adjacency
from metrics.metrics import compute_graph_statistics,create_timestamp_edges,calculate_temporal_katz_index,Edge
import sklearn
import argparse
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import pandas as pd

import glob 
from pathlib import Path
import logging 
import datetime 
import os
import math
from collections import defaultdict
import json

import warnings; warnings.filterwarnings('ignore')

def get_edges_from_adj_graph(graph):
    s = set()
    for start, adj_list in graph.items():
        for end, value in adj_list.items():
            if value > 0:
                start_v, end_v = int(start), int(end)
                start_v, end_v = min(start_v, end_v), max(start_v, end_v)
                s.add("_".join([str(start_v), str(end_v)]))
    return s



logging_info_func = logging.info
def fake_func(msg: object,
    *args: object,
    exc_info=None,
    stack_info: bool = False,
    stacklevel: int = 1,
    extra=None, end=""):
    # for convert to string
    full_msg = str(msg)
    if args:
        full_msg += " " + " ".join([str(item) for item in args])
    if end:
        full_msg += end

    return logging_info_func(full_msg, exc_info=exc_info, stack_info=stack_info, stacklevel=stacklevel, extra=extra)

logging.info = fake_func


parser = argparse.ArgumentParser()
parser.add_argument("--op",help="path of original graphs", type=str)
parser.add_argument("--sp",help="path of sampled graphs", type=str)
parser.add_argument('--time_window',default=1, help="Size of each time window where data needs to be generated",type=int)
parser.add_argument('--debug',default=0, help="debugger",type=int)
parser.add_argument('--config_path',default='', help="path of the config",type=str)
parser.add_argument('--model_name', help="name of the model need to be loaded", type=str)

args = parser.parse_args()


original_graphs_path = args.op
sampled_graphs_path = args.sp
time_window = args.time_window
debug = args.debug
config_path = args.config_path
model_name = args.model_name

#
save_dir = config_path
train_logfilepath = glob.glob(str(Path(save_dir)/ 'train~*~log.txt'))
if len(train_logfilepath) == 0:
    train_logfilepath = glob.glob(str(Path(save_dir)/ '../train~*~log.txt'))[-1]
else: 
    train_logfilepath = train_logfilepath[-1]
    
if not Path(train_logfilepath).exists():
    print("Training log file not found")
    os._exit(os.EX_OK)

project_name = 'ECMLPKDD25_TIGGER'
strtime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
run_name = str(Path(train_logfilepath.replace('train',f'evaluate~{strtime}')).stem)[:-4]

# logfilepath = config_logging(save_dir, run_name, project_name, use_wandb=False)
Path(save_dir).mkdir(parents=True, exist_ok=True)
l_format = logging.Formatter('%(message)s')
logfilepath = Path(save_dir) / f"{run_name}~log.txt"
fhld = logging.FileHandler(logfilepath)
chld = logging.StreamHandler()
fhld.setFormatter(l_format)
chld.setFormatter(l_format)
handlers = [fhld, chld]
logging.basicConfig(level=logging.INFO, handlers=handlers)
logging.info('Log saved in {}'.format(logfilepath))
logging.info(args)
#
#outputfile = args.outputfile
original_graphs = pickle.load(open(Path(config_path) / original_graphs_path,"rb"))
sampled_graphs = pickle.load(open(Path(config_path) / sampled_graphs_path,"rb"))

graph_label_map_path = config_path + "/graph_label_to_id.pkl"
if not Path(graph_label_map_path).exists():
    graph_label_map_path = Path(config_path) / "../graph_label_to_id.pkl"
    
graph_label_to_id = pickle.load(open(graph_label_map_path, "rb"))

id_to_graph_label = {v: k for k, v in graph_label_to_id.items()}
# # Modify the keys of sampled_graphs
# new_keys = {'label0': 0.0, 'label1': 1.0, 'label2': 2.0, 'label3': 3.0, 'label4': 4.0}
# sampled_graphs = {new_keys[key]: sampled_graphs[key] for key in sampled_graphs.keys()}

# # Save the modified sampled_graphs back to the file
# with open(Path(config_path) / sampled_graphs_path, "wb") as f:
#     pickle.dump(sampled_graphs, f)

logging.info("Length of original and sampled,", len(original_graphs), len(sampled_graphs))
commons = defaultdict(dict)
for label in sampled_graphs.keys():
    for snapshot_idx in sampled_graphs[label].keys():
        sgraph = sampled_graphs[label][snapshot_idx]
        ograph = original_graphs[label][snapshot_idx]
        
        sgraphedges = get_edges_from_adj_graph(sgraph)
        ographedges = get_edges_from_adj_graph(ograph)
        len_o = len(ographedges)
        len_common = len(ographedges.intersection(sgraphedges))
        if len_o != 0:
            # commons.append(len_common * 100.0 / len_o)
            commons[label][snapshot_idx] = len_common * 100.0 / len_o

median_abs_error = defaultdict(dict)
mean_abs_error = defaultdict(dict)
for label in commons.keys():
    median_abs_error[label]['%_edge_overlap'] = np.median(list(commons[label].values()))
    mean_abs_error[label]['%_edge_overlap'] = np.mean(list(commons[label].values()))
    
    
old_stats = defaultdict(dict)
new_stats = defaultdict(dict)

for label in sampled_graphs.keys():
    for snapshot_idx in sampled_graphs[label].keys():
        logging.info("\rlabel: %d, snapshot index: %d" % (label, snapshot_idx), end="")
        original_matrix, _, _ = get_numpy_matrix_from_adjacency(original_graphs[label][snapshot_idx])
        sampled_matrix, _, _ = get_numpy_matrix_from_adjacency(sampled_graphs[label][snapshot_idx])

        assert ((original_matrix == original_matrix.T).all())
        assert ((sampled_matrix == sampled_matrix.T).all())
        if original_matrix.shape[0] > 2: 
            if sampled_matrix.shape[0] > 2:
                old_graph_stats = compute_graph_statistics(original_matrix)
                new_graph_stats = compute_graph_statistics(sampled_matrix)
                old_stats[label][snapshot_idx] = old_graph_stats
                new_stats[label][snapshot_idx] = new_graph_stats
                if debug:
                    logging.info("original, sampled,", np.sum(original_matrix > 0) * 0.5, original_matrix.shape[0], np.sum(sampled_matrix > 0) * 0.5, sampled_matrix.shape[0])
            else:
                logging.info("skipping graph with less than 2 nodes, orig: %d, sampled: %d" % (original_matrix.shape[0], sampled_matrix.shape[0]))
            
actual_graph_result = defaultdict(dict)
metrics_to_logs = ['%_edge_overlap', 'd_mean', 'wedge_count', 'triangle_count', 'power_law_exp', 'rel_edge_distr_entropy', 'LCC', 'n_components', 'clustering_coefficient', 'betweenness_centrality_mean', 'closeness_centrality_mean']

for label in old_stats.keys():
    for metric in metrics_to_logs[1:]:
        actual_graph_metrics = [item[metric] for sidx, item in sorted(old_stats[label].items(), key=lambda x: x[0])]
        sampled_graph_metrics = [item[metric] for sidx, item in sorted(new_stats[label].items(), key=lambda x: x[0])]
        abs_error = [abs(a - b) * 1.00 for a, b in zip(actual_graph_metrics, sampled_graph_metrics)]
        infs = [item for item in abs_error if (pd.isnull(item) or math.isinf(item))]
        if len(infs) > 0:
            logging.info("infs found, ", len(infs), metric)
        abs_error = [item for item in abs_error if (not pd.isnull(item) and not math.isinf(item))]
        actual_graph_metrics = [item for item in actual_graph_metrics if (not pd.isnull(item) and not math.isinf(item))]
        median_abs_error[label][metric] = np.median(abs_error)
        actual_graph_result[label][metric] = np.median(actual_graph_metrics)
        mean_abs_error[label][metric] = np.mean(abs_error)
        
logging.info(median_abs_error)

nums = defaultdict(list)
for metric in metrics_to_logs:
    for label in old_stats.keys():
        nums[label].append(median_abs_error[label][metric])

results_dict = {'median_abs_error': median_abs_error, 'mean_abs_error': mean_abs_error, 'actual_median': actual_graph_result}

# Save results to JSON file
# with open(Path(config_path) / sampled_graphs_path.replace('.pkl', '_metrics.json'), 'w') as json_file:
#     json.dump(results_dict, json_file, indent=4)

for label in median_abs_error.keys():
    with open(Path(config_path) / sampled_graphs_path.replace('.pkl', f'_utid_{id_to_graph_label[label]}_metrics.json'), 'w') as json_file:
        saveobj = {'median_abs_error': median_abs_error[label], 'mean_abs_error': mean_abs_error[label], 'actual_median': actual_graph_result[label]}
        json.dump(saveobj, json_file, indent=4)
        print(f"Saved {label} metrics to file")
    
    
for label in nums.keys():
    results = [np.round(item, 4) for item in nums[label]]
    logging.info(f"median for label {label}")
    logging.info(" & ".join([str(item) for item in metrics_to_logs]))
    nums_str = " & ".join(["$" + str(item) + "$" for item in results])
    logging.info(nums_str)


# logging.info("dumping result")









