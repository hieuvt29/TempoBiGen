# adding duration for generated temporal graph
import os
import random
import pandas as pd
import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import random
import pickle
import argparse
import numpy as np
import pickle
import time
import torch
from tqdm import tqdm
from joblib import Parallel, delayed

from model_classes.transductive_model import CondBipartiteEventLSTM

from metrics.metric_utils import (
    get_adj_graph_from_random_walks,
    get_total_nodes_and_edges_from_temporal_adj_list_in_time_range,
    get_adj_origina_graph_from_original_temporal_graph,
    
    sample_adj_graph_multinomial_k_inductive,
    sample_adj_graph_topk,
    
    get_unique_string_from_edge_tuple, 
    get_string_from_edge_tuple, 
    convert_graph_from_defauldict_to_dict
)

from collections import OrderedDict

from tgg_utils import *
import glob 
from pathlib import Path
import logging 
import random
from scipy.stats import lognorm, truncnorm, norm
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

# import warnings
# warnings.filterwarnings("error")

np.int = int
np.float = float

EVAL_LAG = 10 
EPS = 1e-6 

logging_info_func = logging.info


def fake_func(
    msg: object,
    *args: object,
    exc_info=None,
    stack_info: bool = False,
    stacklevel: int = 1,
    extra=None,
    end="",
):
    # for convert to string
    full_msg = str(msg)
    if args:
        full_msg += " " + " ".join([str(item) for item in args])
    if end:
        full_msg += end

    return logging_info_func(
        full_msg,
        exc_info=exc_info,
        stack_info=stack_info,
        stacklevel=stacklevel,
        extra=extra,
    )


logging.info = fake_func

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)



def sequences_from_temporal_walks(generated_events, generated_times, end_node_id, max_t, reverse_vocab, l_w):
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
            if event == end_node_id or time > max_t:
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



def fuse_edges(graph, start_t, end_t, undirected=True):
    ''' 
    fusing multiple temporal edges starting from start_t to end_t into a single edge
    '''
    
    #print(req_deg_seq)
    tedges = {}
    for time in range(start_t, end_t+1):
        if time in graph:
            for start_node, adj_list in graph[time].items():
                for end_node, count in adj_list.items():
                    if start_node != end_node:
                        if undirected:
                            uvt_str = get_unique_string_from_edge_tuple(start_node,end_node,time)
                        else:
                            uvt_str = get_string_from_edge_tuple(start_node,end_node,time)
                        if uvt_str in tedges:
                            tedges[uvt_str] += count
                        else:
                            tedges[uvt_str] = count
    
    #print(tedges)
    fused_edges = {}
    for uvt_str, ct in tedges.items():
        uvt_str = uvt_str.split("#")
        start_node, end_node, orig_t = int(uvt_str[0]), int(uvt_str[1]), float(uvt_str[2])
        uv_str = "{}#{}".format(start_node, end_node)
        if uv_str in fused_edges:
            fused_edges[uv_str][0] += ct
            fused_edges[uv_str][1] += ct * orig_t
        else:
            fused_edges[uv_str] = [ct, ct * orig_t]
    
    return fused_edges
    
def resolve_conflicts(visits, min_gap):
    '''
    Resolve conflicts in visits, conflict is defined as if two visits start time is less than min_gap
    dynamically resolves conflicts so that we keep the maximum number of visits possible
    
    visits: [(room, count, average_fused_start_time, fused_edge_start_time)]
    '''
    if not visits:
        return [], []
    
    resolved_visits = [visits[0]]
    
    if len(visits) == 1:
        return resolved_visits
    
    last_end_time = visits[0][2]  # Initialize with the end time of the first visit
    for visit in visits[1:]:
        room, count, start_time = visit
        if start_time - last_end_time >= min_gap:
            resolved_visits.append(visit)
            last_end_time = start_time
        else:
            # Resolve conflict by keeping the visit with higher count
            if count > resolved_visits[-1][1]:
                resolved_visits[-1] = visit
                last_end_time = start_time
    
    return resolved_visits


def sample_snapshot_src_dst_pairs(adj_matrix_temporal_sampled, min_day, max_day, time_window, 
                                      target_edge_counts, topk_edge_sampling, undirected=True):
    ''' 
    topk_edge_sampling: if True, sample top-k edges with highest counts, 
                        else sample edges based on multinomial distribution with counts as weights
    '''
    sampled_graphs = defaultdict(dict)
    
    for start_time in range(min_day, max_day, time_window):
        snapshot_idx = start_time // time_window
        
        # print("Snapshot idx: {}, start_time: {}".format(snapshot_idx, start_time))

        if target_edge_counts[snapshot_idx] == 0:
            sampled_lb_graph = {}
        elif topk_edge_sampling:
            sampled_lb_graph = sample_adj_graph_topk(
                    adj_matrix_temporal_sampled,
                    start_time,
                    start_time + time_window - 1,
                    target_edge_counts[snapshot_idx],
                    None, None, 
                    undirected
                )
        else:
            sampled_lb_graph = sample_adj_graph_multinomial_k_inductive(
                    adj_matrix_temporal_sampled,
                    start_time,
                    start_time + time_window - 1,
                    target_edge_counts[snapshot_idx],
                    None, None,
                    undirected
                )
        sampled_graphs[snapshot_idx] = sampled_lb_graph
    
    # convert to (hcw, room) pairs
    sampled_hcw_room_pairs = defaultdict(set)
    for snapshot_idx in sampled_graphs:
        sampled_lb_graph = sampled_graphs[snapshot_idx]
        for src_node in sampled_lb_graph:
            for dst_node in sampled_lb_graph[src_node]:
                if is_hcw(vocab[src_node]):
                    hcw, room = src_node, dst_node
                else:
                    assert is_hcw(vocab[dst_node])
                    hcw, room = dst_node, src_node
                    
                sampled_hcw_room_pairs[snapshot_idx].add((hcw, room))
                
    return sampled_hcw_room_pairs, sampled_graphs

def extract_original_snapshot_stats(data, one_day_window, min_t, max_t, undirected=True):
    temporal_graph_original = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    )
    for start, end, day, label in tqdm(data[["start", "end", "days", "label"]].values):
        temporal_graph_original[label][day][start][end] += 1
        if undirected:
            temporal_graph_original[label][day][end][start] += 1
    # convert to dict 
    temporal_graph_original = convert_graph_from_defauldict_to_dict(temporal_graph_original)
        
    original_graphs = defaultdict(dict)
    for label in temporal_graph_original.keys():
        for start_time in range(min_t, max_t, one_day_window):
            snapshot_idx = start_time // one_day_window
            
            og_snapshot = get_adj_origina_graph_from_original_temporal_graph(
                temporal_graph_original[label], start_time, start_time + one_day_window - 1
            )
            original_graphs[label][snapshot_idx] = og_snapshot
            
    orig_daily_hcw_room_pairs = defaultdict(lambda: defaultdict(set))
    for label in original_graphs:
        for snapshot_idx in original_graphs[label]:
            og_snapshot = original_graphs[label][snapshot_idx]
            for src_node in og_snapshot:
                for dst_node in og_snapshot[src_node]:
                    if is_hcw(vocab[src_node]):
                        hcw, room = src_node, dst_node
                    else:
                        assert is_hcw(vocab[dst_node])
                        hcw, room = dst_node, src_node
                    orig_daily_hcw_room_pairs[label][snapshot_idx].add((hcw, room))
    # count number of pairs (src, dst) in each snapshot (not considering the exact time)
    target_daily_node_counts = defaultdict(dict)
    target_daily_edge_counts = defaultdict(dict) 
    for label in tqdm(temporal_graph_original.keys()):
        for start_time in range(min_t, max_t, one_day_window):
            snapshot_idx = start_time // one_day_window
            
            tp, node_count = get_total_nodes_and_edges_from_temporal_adj_list_in_time_range(
                temporal_graph_original[label], start_time, start_time + one_day_window - 1
            )
            if undirected:
                tp = int(tp / 2)
                
            target_daily_edge_counts[label][snapshot_idx] = tp
            target_daily_node_counts[label][snapshot_idx] = node_count

    return target_daily_edge_counts, target_daily_node_counts, orig_daily_hcw_room_pairs


def fuse_events(event_list, fuse_gap):
    '''
    Each event is (cnt, time) in sorted order of time 
    '''
    fused_events = []
    current_group = []
    current_time = None
    
    for event in event_list:
        time = event[1]
        
        if (current_time is None) or (time - current_time <= fuse_gap and time - current_group[0][1] <= fuse_gap):
            # Add to current group
            current_group.append(event)
            current_time = time
        else:
            # Finish current group and start a new one
            if current_group:
                total_count = sum(c for c, _ in current_group)
                avg_time = sum(t * c for c, t in current_group)
                avg_time /= total_count
                fused_events.append((total_count, avg_time))

            current_group = [event]
            current_time = time
            
    # Don't forget the last group
    if current_group:
        total_count = sum(c for c, _ in current_group)
        avg_time = sum(t * c for c, t in current_group)
        avg_time /= total_count
        fused_events.append((total_count, avg_time))
    
    return fused_events

def snapshot_fuse_samepair_edges(graph, fuse_gap, min_t, max_t, is_hcw, vocab, undirected=True):
    '''
    Fuse edges that have same source and destination nodes and temporally close together
    
    Parameters:
    - graph: temporal graph as adjacency list dict[time][src_node][dst_node] = count
    - fuse_gap: maximum time difference for edges to be considered for fusion
    - undirected: whether the graph is undirected
    
    Returns:
    - Dictionary mapping (src, dst) pairs to a list of tuples (count, avg_time)
    '''
    # Extract all events based on (hcw, room) pairs
    nodepair_fused_events = defaultdict(list)
    
    for start_time in range(min_t, max_t, fuse_gap):        
        fused_edges_within_gap = defaultdict(list)
        for time in range(start_time, start_time + fuse_gap):
            if time in graph:
                for start_node in graph[time]:
                    for end_node, count in graph[time][start_node].items():
                        if start_node != end_node:
                            if is_hcw(vocab[start_node]):
                                hcw, room = start_node, end_node
                            else:
                                assert is_hcw(vocab[end_node])
                                hcw, room = end_node, start_node
                            fused_edges_within_gap[(hcw, room)].append((count, time))
        
        # compute avg time and count for each pair
        for (hcw, room), events in fused_edges_within_gap.items():
            total_count = sum(c for c, _ in events)
            avg_time = sum(t * c for c, t in events)
            avg_time /= total_count
            nodepair_fused_events[(hcw, room)].append((total_count, avg_time))
        
    return nodepair_fused_events

def adaptive_fuse_samepair_edges(graph, fuse_gap, is_hcw, vocab, undirected=True):
    '''
    Adaptively fuses edges that have same source and destination nodes and temporally close together
    
    Parameters:
    - graph: temporal graph as adjacency list dict[time][src_node][dst_node] = count
    - fuse_gap: maximum time difference for edges to be considered for fusion
    - undirected: whether the graph is undirected
    
    Returns:
    - Dictionary mapping (src, dst) pairs to a list of tuples (count, avg_time)
    '''
    # Extract all events based on (hcw, room) pairs
    samepair_events = defaultdict(list)
    for time in graph:
        for start_node in graph[time]:
            for end_node, count in graph[time][start_node].items():
                if start_node != end_node:
                    if is_hcw(vocab[start_node]):
                        hcw, room = start_node, end_node
                    else:
                        assert is_hcw(vocab[end_node])
                        hcw, room = end_node, start_node
                        
                    samepair_events[(hcw, room)].append((count, time))
    
    # Sort edges by time
    samepair_events = {k: sorted(v, key=lambda x: x[1]) for k, v in samepair_events.items()}
    
    # Fuse edges that are close in time
    nodepair_fused_events = defaultdict(list)
    for (hcw, room), events in samepair_events.items():
        fused_events = fuse_events(events, fuse_gap)
        nodepair_fused_events[(hcw, room)].extend(fused_events)
        
    return nodepair_fused_events

def post_process_label(data, label, generated_events, generated_times,
                       target_daily_edge_counts, target_daily_nvisits,
                       orig_daily_hcw_room_pairs, vocab, inv_vocab, l_w,
                       min_t, max_t, one_day_window, fuse_gap, fuse_type,
                       end_node_id, is_hcw, pp_save_dir, id_to_graph_label,
                       min_duration, max_duration, hcw_to_cluster,
                       undirected=True, slack_gap=30):
    '''
    label: label to post-process
    generated_events: generated events for the label
    generated_times: generated times for the label
    target_daily_edge_counts: target daily edge counts for the label
    target_daily_nvisits: target daily number of visits for the label
    orig_daily_hcw_room_pairs: original daily hcw-room pairs for the label
    vocab: vocabulary
    inv_vocab: inverse vocabulary
    l_w: length of walk
    min_t: minimum time
    max_t: maximum time
    one_day_window: one day window
    fuse_gap: fuse gap
    undirected: whether the graph is undirected
    '''
    sampled_walks = sequences_from_temporal_walks(generated_events, generated_times,
                                                  end_node_id, max_t, inv_vocab, l_w)
    adj_list_temporal_sampled = get_adj_graph_from_random_walks(sampled_walks, min_t, max_t, True)
    
    # from now on, hcw/room ids are actual ids, not the vocab id
    
    # extract graph with major-voting conflict resolution 
    logging.info(f"\nExtracting graph for label {label} with major-voting conflict resolution")
    
    # Step 1: extract daily-based set of hcw-room pairs
    _, sampled_graphs = sample_snapshot_src_dst_pairs(
        adj_list_temporal_sampled, min_t, max_t, fuse_gap,
        target_daily_edge_counts, topk_edge_sampling=True, 
        undirected=undirected
    )
    
    hcw_room_pairs_graphs = defaultdict(dict)
    for snapshot_idx in sampled_graphs:
        hcw_room_pairs_graphs[snapshot_idx] = defaultdict(int)
        for src in sampled_graphs[snapshot_idx]:
            for dst in sampled_graphs[snapshot_idx][src]:
                if is_hcw(vocab[src]):
                    hcw, room = src, dst
                else:
                    assert is_hcw(vocab[dst])
                    hcw, room = dst, src
                cnt = sampled_graphs[snapshot_idx][src][dst]
                hcw_room_pairs_graphs[snapshot_idx][(hcw, room)] += cnt
            
    hcw_std_graph = defaultdict(list)
    for snapshot_idx in hcw_room_pairs_graphs:
        for hcw, room in hcw_room_pairs_graphs[snapshot_idx]:
            count = hcw_room_pairs_graphs[snapshot_idx][(hcw, room)]
            hcw_std_graph[hcw].append((room, count, snapshot_idx * fuse_gap))
        
    # sort visits based on start time
    for hcw in hcw_std_graph:
        hcw_std_graph[hcw] = sorted(hcw_std_graph[hcw], key=lambda x: x[2])
        
    # resolve conflicts
    hcw_std_graph_resolved = {}
    for hcw in hcw_std_graph:
        resolved_visits = resolve_conflicts(hcw_std_graph[hcw], fuse_gap)
        hcw_std_graph_resolved[hcw] = resolved_visits
    hcw_std_graph = hcw_std_graph_resolved
    
    # compute min, max, average number of visits per HCW
    num_visits_per_hcw = [len(hcw_std_graph[hcw]) for hcw in hcw_std_graph]
    min_visits_per_hcw = min(num_visits_per_hcw)
    max_visits_per_hcw = max(num_visits_per_hcw)
    avg_visits_per_hcw = np.mean(num_visits_per_hcw)
    logging.info(f"Min, max, average number of visits per HCW: {min_visits_per_hcw}, {max_visits_per_hcw}, {avg_visits_per_hcw}")
    hcw_with_one_visits = [hcw for hcw in hcw_std_graph if len(hcw_std_graph[hcw]) == 1]
    logging.info(f"HCWs with only one visit: {hcw_with_one_visits}")
    #
    # get visits count per hcw in original data
    orig_hcw_visits = data[data['label'] == label].groupby('start').size().to_dict()
    num_visits_per_hcw_orig = [orig_hcw_visits[hcw] for hcw in orig_hcw_visits]
    min_visits_per_hcw_orig = min(num_visits_per_hcw_orig)
    max_visits_per_hcw_orig = max(num_visits_per_hcw_orig)
    avg_visits_per_hcw_orig = np.mean(num_visits_per_hcw_orig)
    logging.info(f"Min, max, average number of visits per HCW in original data: {min_visits_per_hcw_orig}, {max_visits_per_hcw_orig}, {avg_visits_per_hcw_orig}")
    logging.info(f"HCWs with only one visit in original data: {[hcw for hcw in orig_hcw_visits if orig_hcw_visits[hcw] == 1]}")            

    # compute visits gap
    hcw_vists_gaps = {}
    for hcw in hcw_std_graph:
        gaps = []
        for v1, v2 in zip(hcw_std_graph[hcw], hcw_std_graph[hcw][1:]):
            gaps.append(v2[2] - v1[2])
        gaps.append(1e10)
        assert np.all(np.array(gaps) > 0)
        hcw_vists_gaps[hcw] = gaps

    # all_gaps = np.array([gap for hcw in hcw_vists_gaps for gap in hcw_vists_gaps[hcw]])
    
    # sample duration from the truncated lognormal mixture
    logging.info(f"Sampling durations for label {label}")
    hcw_std_graph_with_duration = {}
    for hcw in hcw_std_graph:
        ubs = hcw_vists_gaps[hcw]
        ubs = np.array(ubs) - slack_gap  # subtract slack_gap seconds for realistic transition
        ubs = np.minimum(ubs, max_duration)
        assert np.all(ubs > min_duration)
        
        lbs = np.ones_like(ubs) * min_duration
        # uniformly sample durations from the gap
        durations = np.random.uniform(lbs, ubs)
        hcw_std_graph_with_duration[hcw] = [(room, count, start_time, duration) 
                                            for (room, count, start_time), duration 
                                            in zip(hcw_std_graph[hcw], durations)]
    #
    # create dataframes with temporal edges
    hcw_std_graph_resolved_df = []
    for hcw in hcw_std_graph_with_duration:
        assert hcw != 'end_node'
        for visit in hcw_std_graph_with_duration[hcw]:
            room, count, start_time, duration = visit
            hcw_std_graph_resolved_df.append((hcw, room, start_time, id_to_graph_label[label], duration))
    hcw_std_graph_resolved_df = pd.DataFrame(hcw_std_graph_resolved_df, columns="u,i,ts,label,duration".split(","))
    # sort visits by start time
    hcw_std_graph_resolved_df = hcw_std_graph_resolved_df.sort_values(by='ts')
    # save the generated graph
    sampled_graph_savepath = pp_save_dir + f"/sampled_graph_{id_to_graph_label[label]}_{i}.csv"
    hcw_std_graph_resolved_df.to_csv(sampled_graph_savepath, index=False)
    logging.info(f"Sampled graph saved in {sampled_graph_savepath}")
    return hcw_std_graph_resolved_df
    
    
if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='wiki_small', type=str, help='Name of the dataset')
    parser.add_argument("--data_path", help="full path of original dataset in csv format(start,end,time)",
                        type=str)
    parser.add_argument("--gpu_num",help="GPU no. to use, -1 in case of no gpu", type=int)
    parser.add_argument("--config_path",help="full path of the folder where models and related data are saved during training", type=str)
    parser.add_argument("--model_name", help="name of the model need to be loaded", type=str)
    parser.add_argument("--random_walk_sampling_rate", help="No. of epochs to be sampled from random walks",type=int)
    parser.add_argument("--num_of_sampled_graphs",help="No. of times , a graph to be sampled", type=int)
    parser.add_argument("--l_w",default=17,help="lw", type=int)
    parser.add_argument("--seed",default=42,help="seed",type=int)

    parser.add_argument("--lr",default=0.001,help="learning rate", type=float)
    parser.add_argument("--batch_size",default=128,help="batch size", type=int)
    parser.add_argument("--nb_layers",default=2,help="number of layers", type=int)
    parser.add_argument("--nb_lstm_units",default=200,help="number of lstm units", type=int)
    parser.add_argument("--time_emb_dim",default=64,help="time embedding dimension", type=int)
    parser.add_argument("--embedding_dim",default=100,help="embedding dimension", type=int)
    parser.add_argument('--patience', default=50, type=int, help='Patience for early stopping')
    parser.add_argument(
        "--directed", action="store_true", help="Use directed graph"
    )
    parser.add_argument(
        "--fuse_type", default="adaptive", help="Type of fusion to use"
    )
    parser.add_argument(
        "--savedir", default="postpro_best", help="Directory to save post-processed graphs"
    )
    
    args = parser.parse_args()
    
    seed_everything(args.seed)
    dataset_name = args.dataset_name
    data_path = args.data_path
    gpu_num = args.gpu_num
    config_dir = args.config_path
    model_name = args.model_name
    random_walk_sampling_rate  = args.random_walk_sampling_rate
    num_of_sampled_graphs = args.num_of_sampled_graphs
    l_w = args.l_w
    sampling_batch_size = args.batch_size
    fuse_type = args.fuse_type
    savedir = args.savedir
    n_hcw_clusters = 30
    duration_sampling_rate = 2
    one_day_window = 60 * 60 * 24
    slack_gap = 30
    fuse_gap = 60 * 2 # 2 minutes

    # for ftting mixture model for each HCW cluster duration distribution
    max_components, alpha, th = 10, 0.01, 1e-3

    pp_save_dir = str(Path(config_dir) / savedir)
    Path(pp_save_dir).mkdir(parents=True, exist_ok=True)

    strtime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    train_logfilepath = glob.glob(str(Path(config_dir)/ 'train*~log.txt'))[-1]
    if not Path(train_logfilepath).exists():
        logging.info("Training log file not found")
        os._exit(os.EX_OK)

    eval_start_time = time.time()
    run_name = str(Path(train_logfilepath.replace('train',f'postpro~{strtime}')).stem)[:-4]

    logging.info(args)

    strictly_increasing_walks = True
    num_next_edges_to_be_stored = 100
    undirected = not args.directed

    data= pd.read_csv(data_path)
    if 'u' in data.columns:
        data = data.rename(columns={"u": "start", "i": "end", "ts": "days", 'duration': 'd'})
    #
    with open(config_dir + '/nodes_type.pkl', 'rb') as f:
        nodes_idx_to_type, type_to_nodes_idx = pickle.load(f)
    first_par_size = len(type_to_nodes_idx[1])
    
    data = data[["start", "end", "days", "label", "d"]]
    with open(config_dir + '/graph_label_to_id.pkl', 'rb') as f:
        graph_label_to_id = pickle.load(f)
    
    id_to_graph_label = {v: k for k, v in graph_label_to_id.items()}
    
    data['label'] = data['label'].map(graph_label_to_id)
    # start is the id of a hcw, end is the id of a room
    # for each event, find the next visit of the same hcw
    data['next_start'] = data.groupby(["label", "start"])['days'].shift(-1)
    data['next_start'] = data['next_start'].fillna(1e10)
    
    logging.info("Number of unique graph labels", len(data['label'].unique()))
    
    max_days = max(data['days'])
    logging.info("Minimum, maximum timestamps",min(data['days']),max_days)
    data = data.sort_values(by='days',inplace=False)
    logging.info("number of interactions ", data.shape[0])
    
    with open(config_dir+"/vocab.pkl","rb") as f:  
        vocab = pickle.load(f)
        
    print("Loaded vocab from: ", config_dir+"/vocab.pkl")
    inv_vocab = {v: k for k, v in vocab.items()}
    end_node_id = vocab["end_node"]
    
    logging.info("First partition size", first_par_size)
    
    with open(config_dir+"/time_stats.pkl","rb") as f:  
        time_stats = pickle.load(f)
        
    print("Loaded time stats from: ", config_dir+"/time_stats.pkl")
    mean_log_inter_time = time_stats['mean_log_inter_time']
    std_log_inter_time = time_stats['std_log_inter_time']
    
    pad_token = vocab['<PAD>']
    logging.info("Pad token", pad_token)

    if gpu_num == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Computation device, ", device)

    #
    node_embs_savepath = pp_save_dir + "/node_embs.npy"
    if Path(node_embs_savepath).exists():
        node_embs = np.load(node_embs_savepath)
        logging.info("Node embeddings loaded from " + str(node_embs_savepath))
    else:
        batch_size = args.batch_size
        nb_layers = args.nb_layers
        nb_lstm_units = args.nb_lstm_units
        time_emb_dim = args.time_emb_dim
        embedding_dim = args.embedding_dim

        num_labels = len(data['label'].unique())
        
        elstm = CondBipartiteEventLSTM(
            vocab=vocab,
            nb_layers=nb_layers,
            nb_lstm_units=nb_lstm_units,
            time_emb_dim=time_emb_dim,
            embedding_dim=embedding_dim,
            device=device,
            mean_log_inter_time=mean_log_inter_time,
            std_log_inter_time=std_log_inter_time,
            num_labels=num_labels,
            fp_size=first_par_size,
            sp_size=len(vocab) - first_par_size - 2,
        )
        elstm = elstm.to(device)
        num_params = sum(p.numel() for p in elstm.parameters() if p.requires_grad)

        logging.info(f"{elstm}\n")
        logging.info(f'#parameters: {num_params}, '
                    f'{num_params * 4 / 1024:4f} KB, '
                    f'{num_params * 4 / 1024 / 1024:4f} MB.')

        best_model = torch.load(Path(config_dir) / "models/{}.pth".format(model_name), map_location=device)

        # Check if the model was wrapped with DataParallel
        if 'module.' in list(best_model.keys())[0]:
            new_state_dict = OrderedDict()
            for k, v in best_model.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            elstm.load_state_dict(new_state_dict, strict=False)
        else:
            elstm.load_state_dict(best_model, strict=False)

        node_embs = elstm.word_embedding.weight.data.cpu().numpy()
        np.save(node_embs_savepath, node_embs)
        
        logging.info("Node embeddings shape", node_embs.shape)
        logging.info("Node embeddings saved in " + str(pp_save_dir + "/node_embs.npy"))
    
    #
    hcws = list(type_to_nodes_idx[1])
    assert len(hcws) == first_par_size
    assert all([hcw <= first_par_size+1 for hcw in hcws])
    hcws_ids = set(hcws)
    is_hcw = lambda id: id in hcws_ids

    logging.info("HCW embeddings shape", node_embs[hcws].shape)

    fitdist_savepath = pp_save_dir + "/hcw_cluster_dist.pkl"
    
    if Path(fitdist_savepath).exists() and False:
        with open(fitdist_savepath, "rb") as f:
            hcw_gmm = pickle.load(f)
        logging.info("Fitted distributions loaded from " + str(fitdist_savepath))
    else:
        # keep fitting GaussianMixtureModel until the number of samples in each cluster is at least 10
        while True:
            logging.info("Fitting GMM for HCWs embeddings with n_components = ", n_hcw_clusters)
            hcw_gmm = GaussianMixture(n_components=n_hcw_clusters, 
                                    covariance_type='full', 
                                    max_iter=1000, 
                                    random_state=42)
            hcw_gmm.fit(node_embs[hcws])
            # check if the number of samples in each cluster is at least 10
            if np.all(hcw_gmm.weights_ * len(hcws) >= 10):
                break
            n_hcw_clusters -= 1

        # save the fitted distributions
        with open(fitdist_savepath, "wb") as f:
            pickle.dump(hcw_gmm, f)
        logging.info("Fitted distributions saved in " + str(fitdist_savepath))
    
    # predict cluster
    hcw_to_cluster = dict()
    hcw_embs = node_embs[hcws]
    hcw_clusters = hcw_gmm.predict(hcw_embs)
    for hcw, cluster in zip(hcws, hcw_clusters):
        hcw_to_cluster[hcw] = cluster.item()
        
    # viz the durations
    hcw_cluster_durations = defaultdict(list)
    hcw_cluster_upperbounds = defaultdict(list)
    hcw_cluster_nvisits = defaultdict(int)
    cluster_to_hcws = defaultdict(set)
    for i, row in data.iterrows():
        lb, hcw = row['label'], vocab[row['start']]
        duration, gap = row['d'], row['next_start'] - row['days']
        # convert durations, gaps to values, not np.int64
        duration, gap = int(duration), int(gap)
        hcw_cluster = hcw_to_cluster[hcw]
        hcw_cluster_durations[hcw_cluster].append(duration)
        hcw_cluster_upperbounds[hcw_cluster].append(gap)
        hcw_cluster_nvisits[hcw_cluster] += 1
        cluster_to_hcws[hcw_cluster].add(hcw)
    
    cluster_min_gaps = defaultdict(dict)
    cluster_min_duration = defaultdict(dict)
    cluster_max_duration = defaultdict(dict)
    for cluster in hcw_cluster_upperbounds:
        cluster_min_gaps[cluster] = min(hcw_cluster_upperbounds[cluster])
        cluster_min_duration[cluster] = min(hcw_cluster_durations[cluster])
        cluster_max_duration[cluster] = max(hcw_cluster_durations[cluster])

    logging.info('Minimum gaps per cluster', cluster_min_gaps)
    logging.info('Number of visits per cluster', hcw_cluster_nvisits)
    logging.info('Minimum duration per cluster', cluster_min_duration)
    logging.info('Maximum duration per cluster', cluster_max_duration)
    # vizualize 
    #
    min_t = 1
    max_t = int(data['days'].max())

    min_duration = min(cluster_min_duration.values())
    max_duration = max(cluster_max_duration.values())
    logging.info("min_t: {}, max_t: {}".format(min_t, max_t))
    logging.info("Fuse gap: {}, min_duration: {}, max_duration: {}".format(
                    fuse_gap, min_duration, max_duration))
    #

    # get the original temporal graph statistics
    target_snapshot_edge_counts, target_snapshot_node_counts, orig_snapshot_hcw_room_pairs = \
        extract_original_snapshot_stats(data, fuse_gap, min_t, max_t, undirected)
    
    target_snapshot_nvisits = defaultdict(dict)
    # Precompute day indices for all rows in the data
    data['day_idx'] = data['days'] // fuse_gap

    # Group data by label and day_idx for faster access
    grouped_data = data.groupby(['label', 'day_idx']).size()

    for label in target_snapshot_edge_counts.keys():
        for snapshot_idx in target_snapshot_edge_counts[label].keys():
            target_snapshot_nvisits[label][snapshot_idx] = \
                grouped_data.get((label, snapshot_idx), 0)
    # loading the generated graphs (nodes are in vocab, not raw ids)
    for i in range(0, num_of_sampled_graphs):
        print(f"Loading generated events and times for sampled graph {i}")
        with open(config_dir + f"/results_{model_name}/generated_events_{i}.pkl", "rb") as f:
            generated_events = pickle.load(f)
        with open(config_dir + f"/results_{model_name}/generated_times_{i}.pkl", "rb") as f:
            generated_times = pickle.load(f)
        
        # pbar = tqdm(generated_events.keys())
        all_dfs = []
        
        def process_label(label):
            return post_process_label(data, label, generated_events[label], generated_times[label],
                          target_snapshot_edge_counts[label], target_snapshot_nvisits[label],
                          orig_snapshot_hcw_room_pairs[label], vocab, inv_vocab, l_w,
                          min_t, max_t, one_day_window, fuse_gap, fuse_type,
                          end_node_id, is_hcw, pp_save_dir, id_to_graph_label,
                          min_duration, max_duration, hcw_to_cluster,
                          undirected=True, slack_gap=slack_gap)

        all_dfs = Parallel(n_jobs=5)(delayed(process_label)(label) for label in tqdm(generated_events.keys()))
        
        # for label in tqdm(generated_events.keys()):
        #     all_dfs.append(process_label(label))
        
        # concat all csv into one        
        sampled_graph_i = pd.concat(all_dfs, axis=0)
        # sort by timestamp
        sampled_graph_i = sampled_graph_i.sort_values(by='ts', inplace=False)
        sampled_graph_i.to_csv(pp_save_dir + f"/sampled_graph_{i}.csv", index=False)
        logging.info(f"Sampled graph {i} saved in {pp_save_dir + f'/sampled_graph_{i}.csv'}")
