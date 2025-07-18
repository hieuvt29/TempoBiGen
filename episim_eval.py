import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import seaborn as sns
import pickle 


def build_temporal_contact_network(visits, gap = 3600 * 12, importance_factor=10, weighted=True):
    """
    Build a temporal contact network from visit data
    
    Args:
        visits: List of tuples (hid, rid, t, d) representing visits
               hid = healthcare worker ID
               rid = room ID
               t = start time
               d = duration
    
    Returns:
        G: NetworkX graph with weighted edges representing contact time
    """
    # Extract unique HCWs
    hcws = set([visit[0] for visit in visits])
    
    # Track room occupancy over time
    room_occupancy = defaultdict(list)
    
    # Offset the start time of visits to create 0-based time
    min_time = min([visit[2] for visit in visits])
    visits = [(hid, rid, t - min_time, d) for hid, rid, t, d in visits]
    
    # Process visits to determine room occupancy
    for hid, rid, start_time, duration in visits:
        end_time = start_time + duration
        room_occupancy[rid].append((hid, start_time, end_time))
    
    # Calculate contact time between HCWs based on room co-occupancy
    
    snapshots = defaultdict(lambda: defaultdict(float))
    
    for rid, occupancy in room_occupancy.items():
        # For each pair of HCWs in the same room
        for i, (hid1, start1, end1) in enumerate(occupancy):
            for hid2, start2, end2 in occupancy[i+1:]:
                if hid1 == hid2: continue 
                if start2 - end1 > gap: continue
                if start1 - end2 > gap: continue
                
                # Calculate overlapping time
                overlap_start = max(start1, start2)
                overlap_end = min(end1, end2)
                overlap_time = max(0, overlap_end - overlap_start)
                
                key = tuple(sorted([hid1, hid2]))
                snapshot_idx = int(start1 // gap)
                d1 = end1 - start1
                d2 = end2 - start2
                interaction_weight = d1 + d2 + overlap_time * importance_factor
                snapshots[snapshot_idx][key] += interaction_weight
    
    # Devide the weights by the maximum weight
    max_weight = max([max(snap.values()) for snap in snapshots.values()])
    for snapshot_idx in snapshots.keys():
        for key in snapshots[snapshot_idx].keys():
            snapshots[snapshot_idx][key] /= max_weight
            
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    # Create undirected graph for each snapshot, maintaining the node IDs consistent across snapshots
    snapshots_graph = {}
    for snapshot_idx, snapshot in snapshots.items():
        G = nx.Graph()
        for (hid1, hid2), weight in snapshot.items():
            if weighted:
                weight = sigmoid(weight)
            else:
                weight = 1.0
            G.add_edge(hid1, hid2, weight=weight)
        snapshots_graph[snapshot_idx] = G

    return snapshots_graph 
    
def sir_simulation(temporal_graph, node_list, beta, gamma, initial_infected, n_steps, return_details=False):
    # Initialize node states: 0=S, 1=I, 2=R
    time_steps = sorted(temporal_graph.keys())
    N = len(node_list)
    # assert node idx starts from 0 to N-1
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    idx_to_node = {i: node for i, node in enumerate(node_list)}
    
    current_state = np.zeros(N, dtype=int)  # All nodes start as susceptible
    for node in initial_infected:
        current_state[node_to_idx[node]] = 1  # Set initial infected nodes
    
    # Lists to store counts and statuses for this simulation
    sim_S = []
    sim_I = []
    sim_R = []
    sim_status = [] if return_details else None
    
    # Record initial counts
    S_count = np.sum(current_state == 0)
    I_count = np.sum(current_state == 1)
    R_count = np.sum(current_state == 2)
    sim_S.append(S_count)
    sim_I.append(I_count)
    sim_R.append(R_count)
    
    if return_details:
        current_state_dict = {idx_to_node[i]: current_state[i] for i in range(N) if current_state[i] != 0}
        sim_status.append(current_state_dict)
        
    # Simulate over time steps
    for t in range(n_steps):
        G_t = temporal_graph[time_steps[t]]
        new_state = current_state.copy()
        
        # Update node states
        for u in node_list:
            u_status = current_state[node_to_idx[u]]
            eventp = np.random.random_sample()
            # check if u in current snapshot
            if u not in G_t.nodes():
                if u_status == 1 and eventp < gamma:
                    new_state[node_to_idx[u]] = 2
                continue
            
            neighbors = G_t.neighbors(u)
            if isinstance(G_t, nx.DiGraph):
                neighbors = G_t.predecessors(u)

            if u_status == 0:
                infected_neighbors = [(v, G_t[u][v].get('weight', 1.0)) for v in neighbors if current_state[node_to_idx[v]] == 1]
                infection_strength = sum([w for v, w in infected_neighbors])
                if eventp < beta * infection_strength:
                    new_state[node_to_idx[u]] = 1
                    
            elif u_status == 1:
                if eventp < gamma:
                    new_state[node_to_idx[u]] = 2
        
        # Update current state
        current_state = new_state
        
        # Record counts
        S_count = np.sum(current_state == 0)
        I_count = np.sum(current_state == 1)
        R_count = np.sum(current_state == 2)
        sim_S.append(S_count)
        sim_I.append(I_count)
        sim_R.append(R_count)
        
        # Record detailed node statuses if requested
        if return_details:
            current_state_dict = {idx_to_node[i]: current_state[i] for i in range(N) if current_state[i] != 0}
            sim_status.append(current_state_dict)
            
    if return_details:
        return sim_S, sim_I, sim_R, sim_status
    return sim_S, sim_I, sim_R
    
def simulate_sir_temporal_weighted(temporal_graph, node_list, beta, gamma, initial_infected, n_steps, n_sims=10, return_details=False):
    """
    Simulate an SIR model on a temporal weighted graph provided as a dictionary of NetworkX graphs.
    
    Parameters:
    - temporal_graph (dict): Dictionary where keys are time steps and values are nx.Graph objects
                             with weighted edges (weight property).
    - beta (float): Base transmission rate.
    - gamma (float): Recovery rate.
    - initial_infected (list): List of nodes initially infected.
    - n_steps (int): Number of time steps to simulate.
    - n_sims (int): Number of simulations to run (default: 10).
    - return_details (bool): If True, return detailed node statuses over time (default: False).
    
    Returns:
    - S_counts (list): List of lists, susceptible counts over time for each simulation.
    - I_counts (list): List of lists, infected counts over time for each simulation.
    - R_counts (list): List of lists, recovered counts over time for each simulation.
    - node_statuses (list or None): If return_details is True, list of 2D NumPy arrays (n_steps, N)
                                    with node states (0=S, 1=I, 2=R) for each simulation; otherwise None.
    """
    
    # Get sorted time steps from the temporal graph
    time_steps = sorted(temporal_graph.keys())
    n_steps = min(n_steps, len(time_steps))
    
    # Get all node 
    N = len(node_list)
    
    # Initialize output lists
    S_counts = []
    I_counts = []
    R_counts = []
    node_statuses = [] if return_details else None
    
    # Run multiple simulations
    for sim in range(n_sims):
        if return_details:
            S, I, R, status = sir_simulation(temporal_graph, node_list, beta, gamma, initial_infected, n_steps, return_details)
            node_statuses.append(status)
        else:
            S, I, R = sir_simulation(temporal_graph, node_list, beta, gamma, initial_infected, n_steps, return_details)
        
        S_counts.append(S)
        I_counts.append(I)
        R_counts.append(R)
        
    # Return results
    if return_details:
        return S_counts, I_counts, R_counts, node_statuses
    
    return S_counts, I_counts, R_counts


def run_sims(visits, snapshot_gap, beta, gamma, num_init, 
             n_steps, n_sims, weighted=True, savedir=None, run_name=""):   
    temporal_graph = build_temporal_contact_network(visits, gap=snapshot_gap, weighted=weighted)
    # get some statistics from the temporal graph
    print(f"Number of snapshots: {len(temporal_graph)}")
    print(f"Avg number of nodes: {np.mean([G.number_of_nodes() for G in temporal_graph.values()])}")
    print(f"Avg number of edges: {np.mean([G.number_of_edges() for G in temporal_graph.values()])}")
    
    # Simulation parameters
    node_list = list(set([node for G in temporal_graph.values() for node in G.nodes()]))
    N = len(node_list)
    
    initial_infected = np.random.choice(temporal_graph[0].nodes(), num_init, replace=False)
    init_ratio = num_init / N * 100
    print(f"Initial infected nodes: {initial_infected} ({init_ratio:.2f}% of population)")
    # Run simulation with detailed node statuses
    S_counts, I_counts, R_counts, node_statuses = simulate_sir_temporal_weighted(
        temporal_graph, node_list, beta, gamma, initial_infected, n_steps, n_sims, return_details=True
    )
    # Plot average S, I, R counts over time
    t = np.arange(len(S_counts[0]))
    # Compute attack rate as the fraction of non-initial ever-infected nodes at the end of the simulation divided by the total number of nodes
    attack_rates = [(N - S[-1] - num_init) / N for S in S_counts]
    attack_rate_mean = np.mean(attack_rates)
    attack_rate_std = np.std(attack_rates)
    
    # plot the average S, I, R counts over time (with error bars)
    meanS = np.mean(S_counts, axis=0)
    stdS = np.std(S_counts, axis=0)
    meanI = np.mean(I_counts, axis=0)
    stdI = np.std(I_counts, axis=0)
    meanR = np.mean(R_counts, axis=0)
    stdR = np.std(R_counts, axis=0)
    
    #
    plt.figure(figsize=(10, 6))
    plt.plot(t, meanS, label='S', color='blue')
    plt.plot(t, meanI, label='I', color='red')
    plt.plot(t, meanR, label='R', color='green')
    plt.fill_between(t, meanS - stdS, meanS + stdS, color='blue', alpha=0.2)
    plt.fill_between(t, meanI - stdI, meanI + stdI, color='red', alpha=0.2)
    plt.fill_between(t, meanR - stdR, meanR + stdR, color='green', alpha=0.2)
    plt.xlabel('Time Steps')
    plt.ylabel('Node Counts')
    plt.title(f'SIR Simulation (Mean Attack Rate: {attack_rate_mean:.2f} ± {attack_rate_std:.2f})')
    plt.legend()
    plt.grid(True)
    
    savepath = f'{savedir}/sir_simulation_{run_name}_{beta:.2f}_{gamma:.2f}_{num_init}_{weighted}.png'
    plt.savefig(savepath)
    print('saved to ', savepath)
    print(f"Mean attack rate: {attack_rate_mean:.2f} ± {attack_rate_std:.2f}")
    
    return meanS, stdS, meanI, stdI, meanR, attack_rates

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate shifts from generated temporal graphs.')
    parser.add_argument('--opath', type=str, 
                        default='DATA/processed_data/shift2.0_10/all_unittypes_refined.csv',
                        help='Path to original data CSV file')
    parser.add_argument('--bpaths', type=str, nargs='+', 
                        default=['TBG/results/shift2.0_10/postpro_best/sampled_graph_0.csv'],
                        help='List of paths to generated data CSV files')
    parser.add_argument('--bnames', type=str, nargs='+', 
                        default=['bittigger'],
                        help='List of names corresponding to the generated data CSV files')
    
    parser.add_argument('--save_dir', type=str, 
                        default='episim_eval_plots',
                        help='Directory to save evaluation plots')
    
    return parser.parse_args()




if __name__ == "__main__":
    args = parse_arguments()
    save_dir = args.save_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    utids_10 = [v for v in map(int, "0 2 3 5 12 14 21 24 26 28 ".split())]
    utids_5 = [v for v in map(int, "0 2 3 5 12".split())]
    
    utids = utids_10
    
    odata_path = args.opath
    bpaths = args.bpaths
    bnames = args.bnames
    
    all_names = ['ogdata'] + bnames
    all_paths = [odata_path] + bpaths
    
    beta = 0.35  # Base transmission rate
    gamma = 0.2  # Recovery rate
    n_steps = 50  # Number of time steps
    n_sims = 50  # Number of simulations
    snapshot_gap = 3600 * 12  # Snapshot gap in seconds
    num_init = 1 # Number of initially infected nodes in the first snapshot
    savepath = f'{save_dir}/all_sim_results.pkl'
    
    if Path(savepath).exists():
        with open(savepath, 'rb') as f:
            all_sim_results = pickle.load(f)
        print('loaded from ', savepath)
    else:
        all_sim_results = {}
        for name, datapath in zip(all_names, all_paths):
            data = pd.read_csv(datapath)
            data = data[['u', 'i', 'ts', 'duration', 'label']]
            for weighted in [True, False]:
                for utid in utids:
                    print(f"Running simulation for {name} with utid {utid} (weighted={weighted})")
                    data_ut = data[data['label'] == utid][['u', 'i', 'ts', 'duration']]
                    visits = data_ut.values.tolist()
                    
                    meanS, stdS, meanI, \
                    stdI, meanR, attack_rates = \
                        run_sims(visits, snapshot_gap, beta, gamma, 
                                num_init, n_steps, n_sims, weighted, 
                                savedir=save_dir, run_name=f"{utid}_{name}_{weighted}")
                    
                    all_sim_results[(name, utid, weighted)] = (meanS, stdS, meanI, stdI, meanR, attack_rates)
        
        # save all_sim_results
        with open(savepath, 'wb') as f:
            pickle.dump(all_sim_results, f)
        print('saved to ', savepath)
    
    # plot boxplot of attack rates, organized by unit type
    attack_rates_df_5 = pd.DataFrame()
    data_list = []
    for utid in utids_5:
        for name in all_names:                
            for weighted in [True, False]:
                attack_rates = all_sim_results[(name, utid, weighted)][-1]
                utid_str = str(utid)
                if name == 'ogdata': 
                    name_str = 'Real'
                else: 
                    name_str = name
                
                for ar in attack_rates:
                    data_list.append({'key': f"{utid_str}_{name_str}", 
                                    'name': name_str,
                                    'utid': utid_str,
                                    'weighted': weighted, 
                                    'attack_rate': ar})
    attack_rates_df_5 = pd.DataFrame(data_list)
    
    
    # Create a better organized boxplot grouped by unit type
    
    # Use factorplot/catplot to group by unit type first, then by model name
    g = sns.catplot(
        data=attack_rates_df_5,
        kind="box",
        x="name", y="attack_rate",
        hue="weighted",
        col="utid", 
        col_wrap=5,  # Adjust based on number of unit types
        height=5, aspect=0.5,
        sharey=True,
        sharex=False,  # Allow different x-ticks for each subplot
    )
    g._legend.set_title(g._legend.get_title().get_text(), prop={'size': 12})
    for text in g._legend.texts:
        text.set_fontsize(12)
    # Improve the plot appearance
    g.set_axis_labels("", "Attack Rate")
    g.set_xticklabels(rotation=45)
    # Update column titles to display "Utype" instead of "utid"
    for ax, title in zip(g.axes.flat, g.col_names):
        ax.set_title(f"Utype {title}", fontsize=12)
    # Rotate xticks
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    # Set grid for all subplots
    for ax in g.axes.flat:
        ax.grid(True)
    # Make all the text bigger including xtickslabel, ylabel, legend, title
    for ax in g.axes.flat:
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_ylabel(ax.get_ylabel(), fontsize=12)
    # adjust the layout
    g.tight_layout()
    # Save the grouped plot
    savepath = f'{save_dir}/attack_rate_by_unittype.png'
    plt.savefig(savepath)
    print('saved to ', savepath)
    
    # Also create a simpler summary plot with all unit types combined
    attack_rates_df = pd.DataFrame()
    data_list = []
    for utid in utids:
        for name in all_names:                
            for weighted in [True, False]:
                attack_rates = all_sim_results[(name, utid, weighted)][-1]
                utid_str = str(utid)
                if name == 'ogdata': 
                    name_str = 'Real'
                else: 
                    name_str = name
                
                for ar in attack_rates:
                    data_list.append({'key': f"{utid_str}_{name_str}", 
                                    'name': name_str,
                                    'utid': utid_str,
                                    'weighted': weighted, 
                                    'attack_rate': ar})
    attack_rates_df = pd.DataFrame(data_list)
    
    attack_rates_df_summary = attack_rates_df[attack_rates_df['weighted'] == True]
    
    # compute the mean and std of attack rates for each model for each unit type
    attack_rates_means = attack_rates_df_summary.groupby(['utid', 'name']).agg({'attack_rate': 'mean'})
    # compute the differene of attack rates between the real data and the generated data
    attack_rates_means_stds_diff = attack_rates_means - attack_rates_means.xs('Real', level='name')
    # taking absolute value of the difference
    attack_rates_means_stds_diff = attack_rates_means_stds_diff.abs()
    # compute the average difference over all unit types
    attack_rates_means_stds_diff_avg = attack_rates_means_stds_diff.groupby('name').agg({'attack_rate': 'mean'})
    print("Attack Rate Difference average over all unit types: ")
    print(attack_rates_means_stds_diff_avg.to_json())
    
    attack_rates_means_stds_diff_avg_round = attack_rates_means_stds_diff_avg.round(4)
    print(attack_rates_means_stds_diff_avg_round.to_json())
    # exclude "Real" from the attack_rates_means_stds_diff_avg_round
    attack_rates_means_stds_diff_avg_round = attack_rates_means_stds_diff_avg_round.drop('Real')
    
    # Prepare legend labels with average differences
    legend_labels = []
    for name in attack_rates_df_summary['name'].unique():
        if name in attack_rates_means_stds_diff_avg_round.index:
            avg_diff = attack_rates_means_stds_diff_avg_round.loc[name, 'attack_rate']
            legend_labels.append(f"{name} ({avg_diff:.4f})")
        else:
            legend_labels.append(name)

    plt.figure(figsize=(10, 5))
    sns.boxplot(x='utid', y='attack_rate', hue='name', data=attack_rates_df_summary)
    plt.xlabel('Utype')
    plt.ylabel('Attack Rate')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.title("Attack Rate Summary (Weighted)")
    plt.tight_layout()

    # Update legend with modified labels
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles, legend_labels, title="Method (Avg Diff)")

    savepath = f'{save_dir}/attack_rate_summary.png'
    plt.savefig(savepath)
    print('saved to ', savepath)
