import os
import random
import pandas as pd
import datetime
from collections import defaultdict
import numpy as np
import random
import pickle
import argparse
import numpy as np
import pickle
import time
import torch
import h5py
from torch.nn import functional as F

from model_classes.transductive_model import CondEventLSTM
from torch.utils.data import DataLoader, TensorDataset

from collections import OrderedDict

from tgg_utils import *
from train_transductive import seed_everything, config_logging
import glob 
from pathlib import Path
import logging 
import random


### configurations 
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

args = parser.parse_args()


def get_dataloader(sequences, batch_size):
    (
        seq_X, seq_Y, seq_Xt, seq_Yt, seq_XDelta,
        seq_YDelta, seq_labels, X_lengths, Y_lengths, max_len,
    ) = get_X_Y_T_from_sequences(sequences)
    
    dataset = TensorDataset(
        torch.LongTensor(seq_X), torch.LongTensor(seq_Y),
        torch.FloatTensor(seq_Xt), torch.FloatTensor(seq_Yt),
        torch.FloatTensor(seq_XDelta), torch.FloatTensor(seq_YDelta),
        torch.LongTensor(seq_labels), torch.LongTensor(X_lengths),
        torch.LongTensor(Y_lengths)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

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

def run(args):
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

    strtime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    save_dir = config_dir
    train_logfilepath = glob.glob(str(Path(save_dir)/ '*~log.txt'))[-1]
    if not Path(train_logfilepath).exists():
        logging.info("Training log file not found")
        os._exit(os.EX_OK)

    eval_start_time = time.time()
    PROJECT_NAME = "ECMLPKDD25_TIGGER"
    run_name = str(Path(train_logfilepath.replace('train',f'test~{strtime}')).stem)[:-4]

    logfilepath = config_logging(args, save_dir, run_name, PROJECT_NAME, use_wandb=False)
    logging.info(args)

    strictly_increasing_walks = True
    num_next_edges_to_be_stored = 100
    undirected = not args.directed

    data= pd.read_csv(data_path)
    if 'u' in data.columns:
        data = data.rename(columns={"u": "start", "i": "end", "ts": "days"})
        
    data = data[["start", "end", "days", "label"]]
    graph_label_to_id = pickle.load(open(config_dir + '/graph_label_to_id.pkl', 'rb'))
    data['label'] = data['label'].map(graph_label_to_id)
    
    logging.info("Number of unique graph labels", len(data['label'].unique()))
    
    node_set = set(data['start']).union(set(data['end']))
    logging.info("number of nodes,",len(node_set))
    node_set.update('end_node')
    max_days = max(data['days'])
    logging.info("Minimum, maximum timestamps",min(data['days']),max_days)
    data = data.sort_values(by='days',inplace=False)
    logging.info("number of interactions ", data.shape[0])
    
    vocab = pickle.load(open(config_dir+"/vocab.pkl","rb"))
    time_stats = pickle.load(open(config_dir+"/time_stats.pkl","rb"))
    mean_log_inter_time = time_stats['mean_log_inter_time']
    std_log_inter_time = time_stats['std_log_inter_time']

    pad_token = vocab['<PAD>']
    logging.info("Pad token", pad_token)

    hf = h5py.File(config_dir+'/start_node_and_times.h5', 'r')
    start_node_and_times_trained = hf.get('1')
    start_node_and_times_trained = np.array(start_node_and_times_trained)
    start_node_and_times_trained = list(start_node_and_times_trained)
    start_node_and_times_by_label = defaultdict(list)
    for item in start_node_and_times_trained:
        start_node_and_times_by_label[item[0]].append(item[1:])
    
    lb_start_node_sets = {lb: set([item[0] for item in start_node_and_times_by_label[lb]])
                        for lb in start_node_and_times_by_label.keys()}
    
    # logging.info("length of start node and times,", len(start_node_and_times_trained))
    for lb in sorted(start_node_and_times_by_label.keys()):
        logging.info(f"Label {lb}, #lb_edges: ", data[data['label'] == lb].shape[0])
        logging.info(f"Label {lb}, #start_node_and_times: ", len(start_node_and_times_by_label[lb]))
        
    hf.close()

    if gpu_num == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Computation device, ", device)

    lr = args.lr
    batch_size = args.batch_size
    nb_layers = args.nb_layers
    nb_lstm_units = args.nb_lstm_units
    time_emb_dim = args.time_emb_dim
    embedding_dim = args.embedding_dim

    elstm = CondEventLSTM(
        vocab=vocab,
        nb_layers=nb_layers,
        nb_lstm_units=nb_lstm_units,
        time_emb_dim=time_emb_dim,
        embedding_dim=embedding_dim,
        device=device,
        mean_log_inter_time=mean_log_inter_time,
        std_log_inter_time=std_log_inter_time,
        num_labels=len(data['label'].unique())
    )
    elstm = elstm.to(device)
    num_params = sum(p.numel() for p in elstm.parameters() if p.requires_grad)

    logging.info(" ##### Number of parameters#### " ,num_params)

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

    test_save_dir = str(Path(config_dir) / f"results_{model_name}")
    Path(test_save_dir).mkdir(parents=True, exist_ok=True)

    for t in range(0, num_of_sampled_graphs):
        generated_events = {lb: [] for lb in start_node_and_times_by_label.keys()}
        generated_times = {lb: [] for lb in start_node_and_times_by_label.keys()}
        for lb_idx, label in enumerate(sorted(start_node_and_times_by_label.keys())):
            
            start_node_and_times_trained = start_node_and_times_by_label[label]
            num_edges_of_lb = data[data['label'] == label].shape[0]
            sample_size = min(num_edges_of_lb * random_walk_sampling_rate, len(start_node_and_times_trained))
            sampled_start_node_times = random.sample(start_node_and_times_trained, sample_size)
            start_index = 0
            batch_size = sampling_batch_size
            
            logging.info(f"Label {label}, #gen_random_walks: ", len(sampled_start_node_times))

            for start_index in range(0, len(sampled_start_node_times), batch_size):
                if start_index + batch_size < len(sampled_start_node_times):
                    cur_batch_size = batch_size
                else:
                    cur_batch_size = len(sampled_start_node_times) - start_index

                start_node = [[item[0]] for item in sampled_start_node_times[start_index:start_index+cur_batch_size]]
                start_time = [[item[1]] for item in sampled_start_node_times[start_index:start_index+cur_batch_size]]

                batch_X = np.array(start_node)
                batch_Xt = np.array(start_time)
                pad_batch_X = torch.LongTensor(batch_X).to(device)
                pad_batch_Xt = torch.Tensor(batch_Xt).to(device)
                pad_batch_seq_labels = torch.LongTensor([int(label)] * cur_batch_size).to(device)
                #
                label_embs = elstm.conditional_embedding(pad_batch_seq_labels)
                elstm.hidden = elstm.init_hidden(label_embs)
                length = 0
                batch_generated_events = []
                batch_generated_times = []
                batch_generated_events.append(pad_batch_X.detach().cpu().numpy())
                batch_generated_times.append(pad_batch_Xt.detach().cpu().numpy())
                while length < l_w:
                    length += 1    
                    X = pad_batch_X
                    Xt = pad_batch_Xt
                    cur_batch_size, seq_len = X.size() 
                    X = elstm.word_embedding(X)
                    Xt = elstm.t2v(Xt)
                    X = torch.cat((X, Xt), -1)
                    X, elstm.hidden = elstm.lstm(X, elstm.hidden)    
                    X = X.contiguous()
                    X = X.view(-1, X.shape[2])
                    Y_hat = elstm.hidden_to_events(X)
                    Y_hat = F.softmax(Y_hat, dim=-1)
                    sampled_Y = torch.multinomial(Y_hat, 1, replacement=True)### (batch_size*seq_len)*number of replacements
                    sampled_Y = sampled_Y + 1 ### Since event embedding starts from 1, 0 is for padding
                    Y = elstm.word_embedding(sampled_Y)
                    Y = Y.view(-1, Y.shape[-1])
                    X = torch.cat((X, Y), -1)
                    X = elstm.sigmactivation(X)
                    X = elstm.hidden_to_hidden_time(X)
                    X = elstm.sigmactivation(X)
                    X = X.view(cur_batch_size, seq_len, X.shape[-1])  
                    itd = elstm.lognormalmix.get_inter_time_dist(X) #### X is context 
                    T_hat = itd.sample()
                    T_hat = pad_batch_Xt.add(T_hat)
                    T_hat = torch.round(T_hat)

                    pad_batch_X = sampled_Y
                    pad_batch_Xt = T_hat
                    pad_batch_Xt[pad_batch_Xt < 1] = 1
                    batch_generated_events.append(pad_batch_X.detach().cpu().numpy())
                    batch_generated_times.append(pad_batch_Xt.detach().cpu().numpy())

                batch_generated_events = np.array(batch_generated_events).squeeze(-1).transpose()
                batch_generated_times = np.array(batch_generated_times).squeeze(-1).transpose()
                #
                generated_events[label].append(batch_generated_events)
                generated_times[label].append(batch_generated_times)
            
            #
            generated_events[label] = np.concatenate(generated_events[label], axis=0)
            generated_times[label] = np.concatenate(generated_times[label], axis=0)
            gen_node_set = set(generated_events[label].flatten().tolist())
            if len(gen_node_set) < len(lb_start_node_sets[label]):
                logging.info(f"Label {label}, some nodes are not generated, #nodes: {len(gen_node_set)}, #lb_nodes: {len(lb_start_node_sets[label])}")

            elif len(gen_node_set) > len(lb_start_node_sets[label]):
                logging.info(f"Label {label}, some nodes outside the label are generated, #nodes: {len(gen_node_set)}, #lb_nodes: {len(lb_start_node_sets[label])}")
        #
        for lb in sorted(generated_events.keys()):
            logging.info(f"Label {lb}, generated events shape", generated_events[lb].shape)

        # np.save(open(str(Path(test_save_dir) / "generated_events_{}.npy".format(str(t))), "wb"), generated_events)
        # np.save(open(str(Path(test_save_dir) / "generated_times_{}.npy".format(str(t))), "wb"), generated_times)
        pickle.dump(generated_events, open(str(Path(test_save_dir) / "generated_events_{}.pkl".format(str(t))), "wb"))
        pickle.dump(generated_times, open(str(Path(test_save_dir) / "generated_times_{}.pkl".format(str(t))), "wb"))
    
    logging.info("Total time taken {} mins".format((time.time() - eval_start_time) / 60))

if __name__ == "__main__":
    run(args)
