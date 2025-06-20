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
from tqdm import tqdm
import h5py

from model_classes.transductive_model import CondBipartiteEventLSTM
from collections import OrderedDict

from tgg_utils import *
from train_transductive import seed_everything, config_logging
import glob 
from pathlib import Path
import logging 
import random

EPS = 1e-6 

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
    run_name = str(Path(train_logfilepath.replace('train','test')).stem)[:-4]

    logfilepath = config_logging(args, save_dir, run_name, PROJECT_NAME, use_wandb=False)
    logging.info(args)

    strictly_increasing_walks = True
    num_next_edges_to_be_stored = 100
    undirected = not args.directed

    data= pd.read_csv(data_path)
    if 'u' in data.columns:
        data = data.rename(columns={"u": "start", "i": "end", "ts": "days", 'duration': 'd'})
    #
    # mapping_path = Path(data_path).parent / Path(data_path).name.replace(".csv", "_inv_map.pkl")
    # preprocessed_id_mapping = pickle.load(open(mapping_path, "rb"))
    # max_u_id = max(preprocessed_id_mapping["u"].keys())
    with open(config_dir + '/nodes_type.pkl', 'rb') as f:
        nodes_idx_to_type, type_to_nodes_idx = pickle.load(f)
        
    first_par_size = len(type_to_nodes_idx[1])
        
    data = data[["start", "end", "days", "label", "d"]]
    graph_label_to_id = pickle.load(open(config_dir + '/graph_label_to_id.pkl', 'rb'))
    data['label'] = data['label'].map(graph_label_to_id)
    
    logging.info("Number of unique graph labels", len(data['label'].unique()))
    
    max_days = max(data['days'])
    logging.info("Minimum, maximum timestamps",min(data['days']),max_days)
    data = data.sort_values(by='days',inplace=False)
    logging.info("number of interactions ", data.shape[0])
    
    vocab = pickle.load(open(config_dir+"/vocab.pkl","rb"))
    print("Loaded vocab from: ", config_dir+"/vocab.pkl")
    inv_vocab = {v: k for k, v in vocab.items()}
    
    logging.info("First partition size", first_par_size)
    def node_partition(node_idx):
        if node_idx <= 1:
            return 0 
        elif node_idx <= first_par_size + 1:
            return 1
        else:
            return 2
    
    time_stats = pickle.load(open(config_dir+"/time_stats.pkl","rb"))
    print("Loaded time stats from: ", config_dir+"/time_stats.pkl")
    mean_log_inter_time = time_stats['mean_log_inter_time']
    std_log_inter_time = time_stats['std_log_inter_time']
    
    pad_token = vocab['<PAD>']
    logging.info("Pad token", pad_token)

    hf = h5py.File(config_dir+'/start_node_and_times_best.h5', 'r')
    print("Loaded start node and times from: ", config_dir+'/start_node_and_times_best.h5')
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

    # vocab per label 
    vocab_per_label = {}
    for lb in sorted(start_node_and_times_by_label.keys()):
        vocab_per_label[lb] = set(data[data['label'] == lb]['start'].apply(lambda x: vocab[x]).unique())
        vocab_per_label[lb].update(data[data['label'] == lb]['end'].apply(lambda x: vocab[x]).unique())
    
    for lb in sorted(start_node_and_times_by_label.keys()):
        start_nodes_by_lb = [item[0] for item in start_node_and_times_by_label[lb]]
        start_nodes_by_lb = set(start_nodes_by_lb)
        if not len(start_nodes_by_lb.intersection(vocab_per_label[lb])) == len(start_nodes_by_lb):
            logging.info(f"Label {lb}, start nodes not in vocab")
            import pdb; pdb.set_trace()
        
        
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

    test_save_dir = str(Path(config_dir) / f"results_{model_name}")
    Path(test_save_dir).mkdir(parents=True, exist_ok=True)
    lb_keys = start_node_and_times_by_label.keys()
    for t in range(0, num_of_sampled_graphs):

        generated_events = {}
        generated_times = {}
        
        for lb_idx, label in enumerate(sorted(lb_keys)):
            # if lb_idx > 0: break # TODO: remove this line
            generated_events[label] = []
            generated_times[label] = []
            
            start_node_and_times_trained = start_node_and_times_by_label[label]
            num_edges_of_lb = data[data['label'] == label].shape[0]
            sample_size = min(num_edges_of_lb * random_walk_sampling_rate, len(start_node_and_times_trained))
            logging.info(f"Label {label}, #edges * rate: {num_edges_of_lb * random_walk_sampling_rate}, "
                         f"#start_node_and_times: {len(start_node_and_times_trained)}, #sample_size: {sample_size}")
            sampled_start_node_times = random.sample(start_node_and_times_trained, sample_size)
            start_index = 0
            batch_size = sampling_batch_size
            
            logging.info(f"Label {label}, #gen_random_walks: ", len(sampled_start_node_times))
            vocab_by_label = vocab_per_label[label]
            for start_index in tqdm(range(0, len(sampled_start_node_times), batch_size)):
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
                pad_batch_pX = torch.LongTensor([[node_partition(node_idx.item())] for node_idx in batch_X]).to(device)
                pad_batch_seq_labels = torch.LongTensor([int(label)] * cur_batch_size).to(device)
                #
                with torch.no_grad():
                    batch_generated_events, batch_generated_times = elstm.sample(
                        pad_batch_X, pad_batch_Xt, pad_batch_pX, pad_batch_seq_labels, l_w
                    )
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
                import pdb; pdb.set_trace()
            elif len(gen_node_set) > len(lb_start_node_sets[label]):
                logging.info(f"Label {label}, some nodes outside the label are generated, #nodes: {len(gen_node_set)}, #lb_nodes: {len(lb_start_node_sets[label])}")
                
        for lb in sorted(generated_events.keys()):
            logging.info(f"Label {lb}, generated events shape", generated_events[lb].shape)

        # np.save(open(str(Path(test_save_dir) / "generated_events_{}.npy".format(str(t))), "wb"), generated_events)
        # np.save(open(str(Path(test_save_dir) / "generated_times_{}.npy".format(str(t))), "wb"), generated_times)
        pickle.dump(generated_events, open(str(Path(test_save_dir) / "generated_events_{}.pkl".format(str(t))), "wb"))
        pickle.dump(generated_times, open(str(Path(test_save_dir) / "generated_times_{}.pkl".format(str(t))), "wb"))
        logging.info("Total time taken {} mins".format((time.time() - eval_start_time) / 60))

if __name__ == "__main__":
    run(args)
