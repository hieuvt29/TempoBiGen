
import os
import pickle
import pandas as pd
import datetime
from collections import defaultdict
import numpy as np
import random
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tgg_utils import *
import argparse
from model_classes.transductive_model import (
    CondEventLSTM,
    get_event_prediction_rate,
    get_time_mse,
    get_topk_event_prediction_rate,
)
import logging
import wandb
from tqdm import tqdm
from pathlib import Path
import h5py
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import DataParallel
from torch.cuda.amp import GradScaler, autocast

np.int = int
np.float = float
PROJECT_NAME = "ECMLPKDD25_TBG"
USE_WANDB = False
EVAL_LAG = 10 

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


def config_logging(args, save_dir, run_name, project_name=None, use_wandb=False):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    l_format = logging.Formatter("%(message)s")
    logfilepath = Path(save_dir) / f"{run_name}~log.txt"
    fhld = logging.FileHandler(logfilepath)
    chld = logging.StreamHandler()
    fhld.setFormatter(l_format)
    chld.setFormatter(l_format)
    handlers = [fhld, chld]
    logging.basicConfig(level=logging.INFO, handlers=handlers)
    logging.info("Log saved in {}".format(logfilepath))
    #
    if use_wandb:
        wandb.init(
            project=project_name,
            config=vars(args),
            group="training",
            name=run_name,
            mode="online",
        )
        wandb.run.log_code('.')
    return logfilepath


def sample_random_walk_for_edge(edge, l_w, filter_walk):
    walk = run_random_walk_without_temporal_constraints(edge, l_w, 1)
    if walk is None:
        return None
    walk = clean_random_walk(walk)
    if not filter_rw(walk, filter_walk):
        return None
    return (walk, edge.label)

def run(args):
    # USE_WANDB = not args.no_wandb
    data_path = args.data_path
    gpu_num = args.gpu_num
    config_path = args.config_path
    num_epochs = args.num_epochs
    window_interactions = args.window_interactions
    l_w = args.l_w
    filter_walk = args.filter_walk
    lr = args.lr
    batch_size = args.batch_size
    nb_layers = args.nb_layers
    nb_lstm_units = args.nb_lstm_units
    time_emb_dim = args.time_emb_dim
    embedding_dim = args.embedding_dim

    strtime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    save_dir = config_path
    run_id = "~".join(
        map(str, [lr, window_interactions, l_w, filter_walk, num_epochs, args.seed])
    )
    pathid = config_path.split("/")[-1]; assert pathid != ""
    run_name = f"train~tigger~~{pathid}~{run_id}~{args.seed}~{strtime}"
    logfilepath = config_logging(args, save_dir, run_name, PROJECT_NAME, use_wandb=USE_WANDB)
    logging.info("Arguments: {}".format(args))
    seed_everything(args.seed)
    start_time = time.time()

    graph_save_path = os.path.join(config_path, f"input_graph_data_objects.pkl")
    logging.info("Constructing graph objects...")
    # load data
    data = pd.read_csv(data_path)
    if 'u' in data.columns:
        data = data.rename(columns={"u": "start", "i": "end", "ts": "days"})
    
    all_graph_labels = set(data["label"])
    logging.info("Reindexing graph labels")
    graph_label_to_id = {label: i+1 for i, label in enumerate(sorted(all_graph_labels))} # 0 is reserved for padding
    data["label"] = data["label"].apply(lambda x: graph_label_to_id[x])
    logging.info("Number of unique graph labels", len(all_graph_labels))
    pickle.dump(graph_label_to_id, 
                open(config_path + "/graph_label_to_id.pkl", "wb"))
    
    data = data[["start", "end", "days", "label"]]
    node_set = set(data["start"]).union(set(data["end"]))
    logging.info("number of nodes,", len(node_set))
    node_set.update("end_node")
    max_days = max(data["days"])
    logging.info("Minimum, maximum timestamps", min(data["days"]), max_days)
    data = data.sort_values(by="days", inplace=False)
    logging.info("number of interactions,", data.shape[0])

    strictly_increasing_walks = True
    num_next_edges_to_be_stored = 100

    edges = []
    node_id_to_object = {}
    undirected = not args.directed
    
    for start, end, days, label, in data[["start", "end", "days", "label"]].values:
        if start not in node_id_to_object:
            node_id_to_object[start] = Node(id=start, as_start_node=[], as_end_node=[])
        if end not in node_id_to_object:
            node_id_to_object[end] = Node(id=end, as_start_node=[], as_end_node=[])

        edge = Edge(
            start=start, end=end, time=days, outgoing_edges=[], incoming_edges=[], label=label
        )
        edges.append(edge)
        node_id_to_object[start].as_start_node.append(edge)
        node_id_to_object[end].as_end_node.append(edge)
        if undirected:
            edge = Edge(
                start=end, end=start, time=days, outgoing_edges=[], incoming_edges=[], label=label
            )
            edges.append(edge)
            node_id_to_object[end].as_start_node.append(edge)
            node_id_to_object[start].as_end_node.append(edge)
            
    logging.info(
        "length of edges,", len(edges), " length of nodes,", len(node_id_to_object)
    )

    ct = 0
    for edge in tqdm(edges, desc="Building graph objects..."):
        end_node_edges = node_id_to_object[edge.end].as_start_node
        end_node_edges = [e for e in end_node_edges if e.label == edge.label]
        
        index = binary_search_find_time_greater_equal(
            end_node_edges, edge.time, strictly=strictly_increasing_walks
        )
        if index != -1:
            if strictly_increasing_walks:
                edge.outgoing_edges = end_node_edges[
                    index : index + num_next_edges_to_be_stored
                ]
            else:
                edge.outgoing_edges = [
                    item
                    for item in end_node_edges[
                        index : index + num_next_edges_to_be_stored
                    ]
                    if item.end != edge.start
                ]

        start_node_edges = node_id_to_object[edge.start].as_end_node
        start_node_edges = [e for e in start_node_edges if e.label == edge.label]
        
        index = binary_search_find_time_lesser_equal(
            start_node_edges, edge.time, strictly=strictly_increasing_walks
        )
        if index != -1:
            if strictly_increasing_walks:
                edge.incoming_edges = start_node_edges[
                    max(0, index - num_next_edges_to_be_stored) : index + 1
                ]
            else:
                edge.incoming_edges = [
                    item
                    for item in start_node_edges[
                        max(0, index - num_next_edges_to_be_stored) : index + 1
                    ]
                    if item.start != edge.end
                ]
            edge.incoming_edges.reverse()

        ct += 1
        
    for edge in tqdm(edges, desc="Constructing neighbor sampler alias tables..."):
        edge.out_nbr_sample_probs = []
        edge.in_nbr_sample_probs = []

        if len(edge.outgoing_edges) >= 1:
            edge.out_nbr_sample_probs, edge.outJ, edge.outq = (
                prepare_alias_table_for_edge(
                    edge, incoming=False, window_interactions=window_interactions
                )
            )  ### Gaussian Time Sampling

        if len(edge.incoming_edges) >= 1:
            edge.in_nbr_sample_probs, edge.inJ, edge.inq = prepare_alias_table_for_edge(
                edge, incoming=True, window_interactions=2
            )

    vocab = {"<PAD>": 0, "end_node": 1}
    for node in data["start"]:
        if node not in vocab:
            vocab[node] = len(vocab)
    for node in data["end"]:
        if node not in vocab:
            vocab[node] = len(vocab)
            
    logging.info("Graph objects saved to {}".format(graph_save_path))
    
    # number of edges for each labels 
    num_samples_per_label = {lb: data[data["label"] == lb].shape[0] for lb in set(data["label"])}
        
    #
    logging.info("Length of vocab, ", len(vocab))
    logging.info("Id of end node , ",vocab['end_node'])
    pad_token = vocab["<PAD>"]

    num_samples_per_label = {lb: data[data["label"] == lb].shape[0] for lb in set(data["label"])}
    logging.info("Number of samples per label, ", num_samples_per_label)

    #

    def sample_random_Walks():
        # logging.info("Running Random Walk, Length of edges, ", len(edges))
        random_walks = []
        for edge in edges:
            random_walks.append(
                (run_random_walk_without_temporal_constraints(edge, l_w, 1), 
                 edge.label) # edge.label is the graph label
            )
            
        # logging.info("length of collected random walks,", len(random_walks))
        random_walks = [item for item in random_walks if item[0] is not None]
        # logging.info("length of collected random walks after removing None,", len(random_walks))
        random_walks = [(clean_random_walk(item), lb) for item, lb in random_walks]
        random_walks = [(item, lb) for item, lb in random_walks if filter_rw(item, filter_walk)]
        # logging.info("Length of random walks after removing short ranadom walks", len(random_walks))
        return random_walks

    def get_sequences_from_random_walk(random_walks):
        sequences = [(convert_walk_to_seq(item), lb) for item, lb in random_walks]
        sequences = [(convert_seq_to_id(vocab, item), lb) for item, lb in sequences]
        sequences = [(get_time_delta(item, 0), lb) for item, lb in sequences]
        return sequences

    def get_dataloader(sequences, batch_size, nw=4):
        (
            seq_X, seq_Y, seq_Xt, seq_Yt, seq_XDelta,
            seq_YDelta, seq_labels, X_lengths, Y_lengths, max_len,
        ) = get_X_Y_T_from_sequences(sequences)
        
        def pad_or_truncate_sequence(sequence, max_length, pad_token):
            if len(sequence) == max_length:
                return sequence
            elif len(sequence) > max_length:
                return sequence[:max_length]
            else:
                return sequence + [pad_token] * (max_length - len(sequence))

        seq_X = [pad_or_truncate_sequence(seq, max_len, pad_token) for seq in seq_X]
        seq_Y = [pad_or_truncate_sequence(seq, max_len, pad_token) for seq in seq_Y]
        seq_Xt = [pad_or_truncate_sequence(seq, max_len, pad_token) for seq in seq_Xt]
        seq_Yt = [pad_or_truncate_sequence(seq, max_len, pad_token) for seq in seq_Yt]
        seq_XDelta = [pad_or_truncate_sequence(seq, max_len, pad_token) for seq in seq_XDelta]
        seq_YDelta = [pad_or_truncate_sequence(seq, max_len, pad_token) for seq in seq_YDelta]
        
        dataset = TensorDataset(
            torch.LongTensor(seq_X), torch.LongTensor(seq_Y),
            torch.FloatTensor(seq_Xt), torch.FloatTensor(seq_Yt),
            torch.FloatTensor(seq_XDelta), torch.FloatTensor(seq_YDelta),
            torch.LongTensor(seq_labels), torch.LongTensor(X_lengths),
            torch.LongTensor(Y_lengths)
        )

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                                num_workers=nw, pin_memory=True)

        return dataloader

    random_walks = sample_random_Walks()
    sequences = get_sequences_from_random_walk(random_walks)
    # logging.info("Average length of random walks")
    lengths = defaultdict(list)
    for wk, label in random_walks:
        lengths[label].append(len(wk))
    
    logging.info("Length of random walks for each label")
    for label in sorted(lengths.keys()):
        logging.info(f"Label {label}: {np.mean(lengths[label]):.4f} \u00B1 {np.std(lengths[label]):.4f}")

    inter_times = defaultdict(list)
    for seq, label in sequences:
        for item in seq:
            inter_times[label].append(np.log(item[2]))

    mean_log_inter_time = {}
    std_log_inter_time = {}
    logging.info('Mean and std of log inter times: ')
    for label in sorted(inter_times.keys()):
        mean_log_inter_time[label] = np.mean(inter_times[label])
        std_log_inter_time[label] = np.std(inter_times[label])
        logging.info(
            f"Label {label}: {mean_log_inter_time[label]:.4f} \u00B1 {std_log_inter_time[label]:.4f}"
        )
    # get overall mean_log_inter_time and std_log_inter_time
    mean_log_inter_time = np.mean([v for v in mean_log_inter_time.values()])
    std_log_inter_time = np.mean([v for v in std_log_inter_time.values()])
    logging.info(
        f"Overall: {mean_log_inter_time:.4f} \u00B1 {std_log_inter_time:.4f}", end="\n"
    )
    start_node_and_times = defaultdict(list)
    for seq, label in sequences:
        start_node_and_times[label].append((seq[0][0], seq[0][1], seq[0][2]))
    
    config_dir = config_path  ### Change in random walks
    isdir = os.path.isdir(config_dir)
    if not isdir:
        os.mkdir(config_dir)
    isdir = os.path.isdir(config_dir + "/models")
    if not isdir:
        os.mkdir(config_dir + "/models")
    pickle.dump(
        {
            "mean_log_inter_time": mean_log_inter_time,
            "std_log_inter_time": std_log_inter_time,
        },
        open(config_dir + f"/time_stats.pkl", "wb"),
    )
    pickle.dump(vocab, open(config_dir + f"/vocab.pkl", "wb"))

    def evaluate_model(elstm, dataloader):
        elstm.eval()
        running_loss = []
        running_event_loss = []
        running_time_loss = []
        event_prediction_rates = []
        top5_event_prediction_rates = []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                pad_batch_X, pad_batch_Y, pad_batch_Xt, pad_batch_Yt, \
                pad_batch_XDelta, pad_batch_YDelta, pad_batch_seq_labels, \
                batch_X_len, batch_Y_len = [x.to(device) for x in batch]
                
                mask_distribution = pad_batch_Y != 0
                num_events_time_loss = mask_distribution.sum().item()

                Y_hat, inter_time_log_loss = elstm.forward(
                    X=pad_batch_X,
                    Y=pad_batch_Y,
                    Xt=pad_batch_Xt,
                    Yt=pad_batch_Yt,
                    XDelta=pad_batch_XDelta,
                    YDelta=pad_batch_YDelta,
                    seq_labels=pad_batch_seq_labels,
                    X_lengths=batch_X_len,
                    mask=mask_distribution,
                )

                Y = pad_batch_Y.view(-1) - 1
                Y_hat = Y_hat.view(-1, Y_hat.shape[-1])
                inter_time_log_loss *= mask_distribution
                loss_time = (-1) * inter_time_log_loss.sum() / num_events_time_loss
                loss_event = celoss(Y_hat, Y)
                loss = loss_event + loss_time

                running_loss.append(loss.item())
                running_event_loss.append(loss_event.item())
                running_time_loss.append(loss_time.item())
                event_prediction_rates.append(get_event_prediction_rate(Y, Y_hat))
                top5_event_prediction_rates.append(get_topk_event_prediction_rate(Y, Y_hat, k=5))

        logging.info(f"Val all loss: {np.mean(running_loss):.6f}")
        logging.info(f"Val event loss: {np.mean(running_event_loss):.6f} ± {np.std(running_event_loss):.6f}")
        logging.info(f"Val time loss: {np.mean(running_time_loss):.6f} ± {np.std(running_time_loss):.6f}")
        logging.info(f"Val Event prediction rate: {np.mean(event_prediction_rates):.6f}")
        logging.info(f"Val Event prediction rate@top5: {np.mean(top5_event_prediction_rates):.6f}")
        if USE_WANDB: wandb.log({
            "val_all_loss": np.mean(running_loss),
            "val_event_loss": np.mean(running_event_loss),
            "val_time_loss": np.mean(running_time_loss),
            "event_prediction_rate": np.mean(event_prediction_rates),
            "val_top5_event_pred_rate": np.mean(top5_event_prediction_rates),
        })
        return np.mean(top5_event_prediction_rates)

    def get_X_Y_T_from_sequences(sequences):
        seq_X = []
        seq_Y = []
        seq_Xt = []
        seq_Yt = []
        seq_XDelta = []
        seq_YDelta = []
        seq_labels = []
        
        for seq, lb in sequences:
            seq_X.append([item[0] for item in seq[:-1]])  ## O contain node id
            seq_Y.append([item[0] for item in seq[1:]])
            seq_Xt.append([item[1] for item in seq[:-1]])  ## 1 contain timestamp
            seq_Yt.append([item[1] for item in seq[1:]])
            seq_XDelta.append(
                [item[2] for item in seq[:-1]]
            )  ## 2 contain delta from previous event
            seq_YDelta.append([item[2] for item in seq[1:]])
            seq_labels.append(lb)
            
        X_lengths = [len(sentence) for sentence in seq_X]
        Y_lengths = [len(sentence) for sentence in seq_Y]
        X_labels = [lb for lb in seq_labels]
        
        max_len = max(X_lengths)
        return (
            seq_X,
            seq_Y,
            seq_Xt,
            seq_Yt,
            seq_XDelta,
            seq_YDelta,
            X_labels,
            X_lengths,
            Y_lengths,
            max_len,
        )

    def get_batch(
        start_index,
        batch_size,
        seq_X,
        seq_Y,
        seq_Xt,
        seq_Yt,
        seq_XDelta,
        seq_YDelta,
        X_labels,
        X_lengths,
        Y_lengths,
    ):
        batch_X = seq_X[start_index : start_index + batch_size]
        batch_Y = seq_Y[start_index : start_index + batch_size]
        batch_Xt = seq_Xt[start_index : start_index + batch_size]
        batch_Yt = seq_Yt[start_index : start_index + batch_size]
        batch_XDelta = seq_XDelta[start_index : start_index + batch_size]
        batch_YDelta = seq_YDelta[start_index : start_index + batch_size]
        batch_X_len = X_lengths[start_index : start_index + batch_size]
        batch_Y_len = Y_lengths[start_index : start_index + batch_size]
        batch_X_labels = X_labels[start_index : start_index + batch_size]
        max_len = max(batch_X_len)

        pad_batch_X = np.ones((batch_size, max_len), dtype=np.int64) * pad_token
        pad_batch_Y = np.ones((batch_size, max_len), dtype=np.int64) * pad_token
        pad_batch_Xt = np.ones((batch_size, max_len), dtype=np.float32) * pad_token
        pad_batch_Yt = np.ones((batch_size, max_len), dtype=np.float32) * pad_token
        pad_batch_XDelta = np.ones((batch_size, max_len), dtype=np.float32) * pad_token
        pad_batch_YDelta = np.ones((batch_size, max_len), dtype=np.float32) * pad_token
        pad_batch_seq_labels = np.ones((batch_size), dtype=np.int64) * pad_token

        for i, x_len in enumerate(batch_X_len):
            pad_batch_X[i, 0:x_len] = batch_X[i][:x_len]
            pad_batch_Y[i, 0:x_len] = batch_Y[i][:x_len]
            pad_batch_Xt[i, 0:x_len] = batch_Xt[i][:x_len]
            pad_batch_Yt[i, 0:x_len] = batch_Yt[i][:x_len]
            pad_batch_XDelta[i, 0:x_len] = batch_XDelta[i][:x_len]
            pad_batch_YDelta[i, 0:x_len] = batch_YDelta[i][:x_len]
            pad_batch_seq_labels[i] = batch_X_labels[i]

        pad_batch_X = torch.LongTensor(pad_batch_X).to(device)
        pad_batch_Y = torch.LongTensor(pad_batch_Y).to(device)
        pad_batch_Xt = torch.Tensor(pad_batch_Xt).to(device)
        pad_batch_Yt = torch.Tensor(pad_batch_Yt).to(device)
        pad_batch_XDelta = torch.Tensor(pad_batch_XDelta).to(device)
        pad_batch_YDelta = torch.Tensor(pad_batch_YDelta).to(device)
        pad_batch_seq_labels = torch.LongTensor(pad_batch_seq_labels).to(device)
        
        batch_X_len = torch.LongTensor(batch_X_len).to(device)
        batch_Y_len = torch.LongTensor(batch_Y_len).to(device)
        return (
            pad_batch_X,
            pad_batch_Y,
            pad_batch_Xt,
            pad_batch_Yt,
            pad_batch_XDelta,
            pad_batch_YDelta,
            pad_batch_seq_labels,
            batch_X_len,
            batch_Y_len,
        )

    def data_shuffle(
        seq_X, seq_Y, seq_Xt, seq_Yt, seq_XDelta, seq_YDelta, seq_labels, X_lengths, Y_lengths
    ):
        indices = list(range(0, len(seq_X)))
        random.shuffle(indices)
        #### Data Shuffling
        seq_X = [seq_X[i] for i in indices]  ####
        seq_Y = [seq_Y[i] for i in indices]
        seq_Xt = [seq_Xt[i] for i in indices]
        seq_Yt = [seq_Yt[i] for i in indices]
        seq_XDelta = [seq_XDelta[i] for i in indices]
        seq_YDelta = [seq_YDelta[i] for i in indices]
        seq_labels = [seq_labels[i] for i in indices]
        X_lengths = [X_lengths[i] for i in indices]
        Y_lengths = [Y_lengths[i] for i in indices]
        return (
            seq_X,
            seq_Y,
            seq_Xt,
            seq_Yt,
            seq_XDelta,
            seq_YDelta,
            seq_labels,
            X_lengths,
            Y_lengths,
        )

    (
        seq_X,
        seq_Y,
        seq_Xt,
        seq_Yt,
        seq_XDelta,
        seq_YDelta,
        seq_labels,
        X_lengths,
        Y_lengths,
        max_len,
    ) = get_X_Y_T_from_sequences(sequences)
    
    logging.info("Max lengths of walks", max_len)
    
    seq_X, seq_Y, seq_Xt, seq_Yt, seq_XDelta, seq_YDelta, seq_labels, X_lengths, Y_lengths = (
        data_shuffle(
            seq_X, seq_Y, seq_Xt, seq_Yt, seq_XDelta, seq_YDelta, seq_labels, X_lengths, Y_lengths
        )
    )
    #
    
    if gpu_num == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Computation device, ", device)

    elstm = CondEventLSTM(
        vocab=vocab,
        nb_layers=nb_layers,
        nb_lstm_units=nb_lstm_units,
        time_emb_dim=time_emb_dim,
        embedding_dim=embedding_dim,
        device=device,
        mean_log_inter_time=mean_log_inter_time,
        std_log_inter_time=std_log_inter_time,
        num_labels=len(set(all_graph_labels))
    )
    
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        logging.info(f"Using {num_gpus} GPUs")
        elstm = DataParallel(elstm)

    elstm = elstm.to(device)
    celoss = nn.CrossEntropyLoss(ignore_index=-1)  #### -1 is padded
    optimizer = optim.Adam(elstm.parameters(), lr=lr)  # .001
    scaler = GradScaler()
    num_params = sum(p.numel() for p in elstm.parameters() if p.requires_grad)
    
    logging.info(f"{elstm}\n")
    logging.info(f'#parameters: {num_params}, '
                f'{num_params * 4 / 1024:4f} KB, '
                f'{num_params * 4 / 1024 / 1024:4f} MB.')

    # elstm.train()
    print_ct = 100000
    wt_update_ct = 0
    patience_cnt = 0
    best_loss = 1e10
    pbar = tqdm(range(0, num_epochs + 1), desc="Training...")
    for epoch in pbar:
        elstm.train()
        running_loss = []
        running_event_loss = []
        running_time_loss = []
        # event_prediction_rates = []
        # topk_event_prediction_rates = []
        # topk10_event_prediction_rates = []
        
        # Sample random walks using either parallel or sequential approach
        start_sampling = time.time()
        random_walks = sample_random_Walks()
        sampling_time = time.time() - start_sampling
        logging.info(f"Random walk sampling took {sampling_time:.2f} seconds")
        
        sequences = get_sequences_from_random_walk(random_walks)
        dataloader = get_dataloader(sequences, batch_size, nw=num_gpus)

        accum_init_seq_lbs = set()
        for lb in num_samples_per_label.keys():
            if len(start_node_and_times[lb]) < num_samples_per_label[lb] * 10:
                accum_init_seq_lbs.add(lb)
                
        for seq, label in sequences:
            if label in accum_init_seq_lbs:
                start_node_and_times[label].append((seq[0][0], seq[0][1], seq[0][2]))

        
        for batch in dataloader:
            pad_batch_X, pad_batch_Y, pad_batch_Xt, pad_batch_Yt, \
            pad_batch_XDelta, pad_batch_YDelta, pad_batch_seq_labels, \
            batch_X_len, batch_Y_len = [x.to(device, non_blocking=True) for x in batch]

            elstm.zero_grad()
            mask_distribution = pad_batch_Y != 0
            num_events_time_loss = mask_distribution.sum().item()

            with autocast(enabled=False):
                Y_hat, inter_time_log_loss = elstm.forward(
                    X=pad_batch_X,
                    Y=pad_batch_Y,
                    Xt=pad_batch_Xt,
                    Yt=pad_batch_Yt,
                    XDelta=pad_batch_XDelta,
                    YDelta=pad_batch_YDelta,
                    seq_labels=pad_batch_seq_labels,
                    X_lengths=batch_X_len,
                    mask=mask_distribution,
                )

                inter_time_log_loss *= mask_distribution
                loss_time = (-1) * inter_time_log_loss.sum() / num_events_time_loss
                Y = pad_batch_Y.view(-1) - 1
                Y_hat = Y_hat.view(-1, Y_hat.shape[-1])
                loss_event = celoss(Y_hat, Y)
                loss = loss_event + loss_time

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(elstm.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss.append(loss.item())
            running_event_loss.append(loss_event.item())
            running_time_loss.append(loss_time.item())

        if USE_WANDB: wandb.log({
            "train_all_loss": np.mean(running_loss),
            "train_event_loss": np.mean(running_event_loss),
            "train_time_loss": np.mean(running_time_loss),
        })

        if epoch % EVAL_LAG == 0 or epoch == num_epochs:
            # log the time now 
            logging.info(f"@Epoch {epoch} Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logging.info(f"Train all loss: {np.mean(running_loss):.6f}")
            logging.info(
                f"Train event loss: {np.mean(running_event_loss):.6f} ± {np.std(running_event_loss):.6f}"
            )
            logging.info(
                f"Running time loss: {np.mean(running_time_loss):.6f} ± {np.std(running_time_loss):.6f}"
            )
            logging.info("Running evaluation")
            perf = evaluate_model(elstm, get_dataloader(get_sequences_from_random_walk(
                                            sample_random_Walks()), batch_size, nw=num_gpus))
            torch.save(
                elstm.state_dict(), config_dir + f"/models/{str(epoch)}.pth"
            )
            
            if len(accum_init_seq_lbs) > 0:
                start_node_and_times_list = [(label, start, time, delta) 
                                                for label, values in start_node_and_times.items() 
                                                for start, time, delta in values]

                hf = h5py.File(config_dir + f"/start_node_and_times.h5", "w")
                hf.create_dataset("1", data=start_node_and_times_list)
                hf.close()
        

        # add early stoping based on np.mean(running_loss)
        avg_running_loss = np.mean(running_loss)
        if avg_running_loss < best_loss:
            best_loss = avg_running_loss
            patience_cnt = 0
            torch.save(elstm.state_dict(), config_dir + "/models/best.pth")
            logging.info("Best model saved")
            if len(accum_init_seq_lbs) > 0:
                start_node_and_times_list = [(label, start, time, delta) 
                                                for label, values in start_node_and_times.items() 
                                                for start, time, delta in values]
                    
                hf = h5py.File(config_dir + "/start_node_and_times_best.h5", "w")
                hf.create_dataset("1", data=start_node_and_times_list)
                hf.close()
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                logging.info(f"Early stopping at epoch {epoch}")
                break
        pbar.set_description(f"Epoch {epoch}/{num_epochs}: Loss {avg_running_loss:.4f}")
        
    logging.info("Total time taken for training", time.time() - start_time)
    logging.info("Log saved to ", logfilepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", default="wiki_small", type=str, help="Name of the dataset"
    )
    parser.add_argument(
        "--data_path",
        help="full path of dataset in csv format(start,end,time)",
        type=str,
    )
    parser.add_argument(
        "--gpu_num", help="GPU no. to use, -1 in case of no gpu", type=int, default=0
    )
    parser.add_argument(
        "--config_path",
        help="full path of the folder where models and related data need to be saved",
        type=str,
    )
    parser.add_argument(
        "--num_epochs", default=200, help="Number of epochs for training", type=int
    )
    parser.add_argument(
        "--window_interactions", default=6, help="Interaction window", type=int
    )
    parser.add_argument("--l_w", default=20, help="lw", type=int)
    parser.add_argument("--filter_walk", default=2, help="filter_walk", type=int)
    parser.add_argument("--seed", default=0, help="seed", type=int)

    parser.add_argument("--lr", default=0.001, help="learning rate", type=float)
    parser.add_argument("--batch_size", default=128, help="batch size", type=int)
    parser.add_argument("--nb_layers", default=2, help="number of layers", type=int)
    parser.add_argument(
        "--nb_lstm_units", default=200, help="number of lstm units", type=int
    )
    parser.add_argument(
        "--time_emb_dim", default=64, help="time embedding dimension", type=int
    )
    parser.add_argument(
        "--embedding_dim", default=100, help="embedding dimension", type=int
    )
    parser.add_argument(
        "--patience", default=50, type=int, help="Patience for early stopping"
    )
    parser.add_argument(
        "--no_wandb", action="store_true", help="Disable wandb logging"
    )
    parser.add_argument(
        "--directed", action="store_true", help="Use directed graph"
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Use parallel processing for random walk sampling"
    )
    parser.add_argument(
        "--num_workers", default=4, type=int, help="Number of parallel workers for random walk sampling"
    )
    args = parser.parse_args()
    run(args)
