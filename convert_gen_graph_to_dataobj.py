import argparse
import pickle
import torch
from torch_geometric.data import TemporalData

# Argument parser
parser = argparse.ArgumentParser(description='Convert generated graph to TemporalData object.')
parser.add_argument('--opath', type=str, required=True, help='Path to the original graphs pickle file.')
parser.add_argument('--gpath', type=str, required=True, help='Path to the generated graphs pickle file.')
parser.add_argument('--label_path', type=str, required=True, help='Path to the graph label to id pickle file.')
parser.add_argument('--time_window', type=int, default=86400, help='Time window for the temporal edges.')
args = parser.parse_args()

# load paths from arguments
opath = args.opath
gpath = args.gpath
label_path = args.label_path

odata = pickle.load(open(opath, 'rb'))
gdata = pickle.load(open(gpath, 'rb'))
time_window = args.time_window

graph_label_to_id = pickle.load(open(label_path, 'rb'))
id_to_label = {v: k for k, v in graph_label_to_id.items()}

for utid in gdata.keys():
    utid_orig_graph = odata[utid]
    utid_gen_graph = gdata[utid]
    
    og_temporal_edges = []
    for day in utid_orig_graph.keys():
        daily_o = utid_orig_graph[day]
        for s in daily_o:
            for d in daily_o[s]:
                og_temporal_edges.append((s, d, day * time_window))

    # create temporal edges for the sampled graph
    gg_temporal_edges = []
    for day in utid_gen_graph.keys():
        daily_g = utid_gen_graph[day]
        for s in daily_g:
            for d in daily_g[s]:
                gg_temporal_edges.append((s, d, day * time_window))


    # %%
    og_src = torch.tensor(list([int(e[0]) for e in og_temporal_edges]), dtype=torch.long)
    og_dst = torch.tensor(list([int(e[1]) for e in og_temporal_edges]), dtype=torch.long)
    og_t = torch.tensor(list([int(e[2]) for e in og_temporal_edges]), dtype=torch.long)
    print("Original temporal time from {} to {}".format(og_t.min(), og_t.max()))
    
    gg_src = torch.tensor(list([int(e[0]) for e in gg_temporal_edges]), dtype=torch.long)
    gg_dst = torch.tensor(list([int(e[1]) for e in gg_temporal_edges]), dtype=torch.long)
    gg_t = torch.tensor(list([int(e[2]) for e in gg_temporal_edges]), dtype=torch.long)
    print("Generated temporal time from {} to {}".format(gg_t.min(), gg_t.max()))
    # create TemporalData object 
    og_data = TemporalData(src=og_src, dst=og_dst, t=og_t)
    gg_data = TemporalData(src=gg_src, dst=gg_dst, t=gg_t)
    
    # save the data as pt files
    og_fname = opath.replace('.pkl', f'_utid_{id_to_label[utid]}.pt')
    gg_fname = gpath.replace('.pkl', f'_utid_{id_to_label[utid]}.pt')
    
    torch.save(og_data, og_fname)
    torch.save(gg_data, gg_fname)
    print(f"Saved {og_fname}")
    print(f"Saved {gg_fname}")