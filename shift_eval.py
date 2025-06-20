import pickle
import torch 
import datetime
from datetime import timedelta
import numpy as np
import json 
from collections import defaultdict
from pathlib import Path
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

time_window = 86400
START_ITIME = '2024-01-01 00:00:00'

def shift_construction(cfg, visits, hid2jtid_map=None):
    ''' 
    Taking a list of visit events with the following fields:
    (utid, hid, rid, itime, duration), construct shifts by grouping visits by hcw
    (hid) and time (itime).  Assumed that the visits are sorted by itime.
    
    Shifts are defined as contiguous visits
    by the same hcw with no more than cfg['shgap'] seconds between
    them.  Shifts are tagged with a unique shift identifier, and
    augmented with visit count, cumulative visit duration, and
    average visit duration.  Shifts are also tagged with a shift type
    (day or night) based on the time of day.  Shifts are then output
    to file.sft, and the visit data is augmented with the shift
    identifier and output to file.csv.  Finally, some summary
    statistics are output to file.dat.
    '''
    # compute the otime by adding the duration to the itime
    if 'otime' not in visits[0]:
        for visit in visits:
            visit['otime'] = visit['itime'] + visit['duration']
    
    # add a unique visit id if not available
    if 'vid' not in visits[0]:
        for i, visit in enumerate(visits):
            visit['vid'] = i
    
    # add a job type id (map from hid) if not available
    if 'jtid' not in visits[0] and hid2jtid_map is not None:
        for visit in visits:
            visit['jtid'] = hid2jtid_map[visit['hid']]
    
    # convert the itime and otime to datetime objects
    for visit in visits:
        visit['itime'] = datetime.datetime.fromtimestamp(visit['itime'])
        visit['otime'] = datetime.datetime.fromtimestamp(visit['otime'])
    
    lastShift = {}   # key is hid, stores entire visit (a row from the file)
    newShifts = []   # records shifts

    #
    nshifts = 0
    for row in visits:
        if row['hid'] not in lastShift: 
            # No previous visit by hcw hid, so by definition a new
            # shift.
            nshifts = nshifts + 1
            # Save this shift in the lastShift dictionary, indexed by
            # the hid, while tagging the original row copy as well as
            # the shift record with the shift identifier.
            row['shid'] = nshifts
            lastShift[row['hid']] = row.copy()
            # In addition to shid, we augment the initial row entry
            # with visit count and cumulative visit duration, stored,
            # for now, in the average duration slot.
            lastShift[row['hid']]['shid'] = nshifts
            lastShift[row['hid']]['vcount'] = 1
            lastShift[row['hid']]['avgvdur'] = int(row['duration'])
            lastShift[row['hid']]['rcount'] = set([row['rid']])
            lastShift[row['hid']]['itime'] = row['itime']
            lastShift[row['hid']]['otime'] = row['otime']
            lastShift[row['hid']]['duration'] = int((row['otime'] - row['itime']).total_seconds())
        elif int((row['itime'] - lastShift[row['hid']]['otime']).total_seconds()) <= cfg['shgap']:
            # This is a new visit by hcw hid within maxgap time
            # interval: tag the visit with the shid, and update the
            # shift's current counters.
            if row['itime'] < lastShift[row['hid']]['otime']:
                print("HCW appears at two places at the same time.")
                import pdb; pdb.set_trace()
            
            row['shid'] = lastShift[row['hid']]['shid']
            lastShift[row['hid']]['vcount'] = lastShift[row['hid']]['vcount'] + 1
            lastShift[row['hid']]['avgvdur'] = lastShift[row['hid']]['avgvdur']+int(row['duration'])
            lastShift[row['hid']]['rcount'].add(row['rid'])
            # Update existing data, extending duration.
            lastShift[row['hid']]['otime'] = row['otime']
            lastShift[row['hid']]['duration'] = int((row['otime'] - lastShift[row['hid']]['itime']).total_seconds())
            
        else: 
            # Previous shift was too long ago: close current shift
            # and, once everything is updated, start a new one.
            stime = lastShift[row['hid']]['itime']
            etime = lastShift[row['hid']]['otime']
            # Night shifts either span midnight, or begin very early
            # in the morning (before 5am) and ends by 8am (the latter
            # is quite arbitrary).
            if (stime.day != etime.day) or (stime.hour >= 18 and etime.hour < 24) or (stime.hour < 5 and etime.hour < 8):
                lastShift[row['hid']]['shift'] = 'night'
            else:
                lastShift[row['hid']]['shift'] = 'day'

            # Calculate average visit duration (overwrites cumulative visit duration stored in avgvdur)
            lastShift[row['hid']]['avgvdur'] = lastShift[row['hid']]['avgvdur']/(60*lastShift[row['hid']]['vcount'])
            # Calculate number of unique rooms (overwrites set of rids)
            lastShift[row['hid']]['rcount'] = len(lastShift[row['hid']]['rcount'])

            # Append the closed shift to the new shift list.
            newShifts.append(lastShift[row['hid']])

            # Start the new shift.
            nshifts = nshifts + 1
            row['shid'] = nshifts
            lastShift[row['hid']] = row.copy()
            lastShift[row['hid']]['vcount']=1
            lastShift[row['hid']]['avgvdur'] = int(row['duration'])
            lastShift[row['hid']]['rcount'] = set([row['rid']])
            lastShift[row['hid']]['itime'] = row['itime']
            lastShift[row['hid']]['otime'] = row['otime']
            lastShift[row['hid']]['duration'] = int((row['otime'] - row['itime']).total_seconds())
            del lastShift[row['hid']]['rid']
            
    # Flush any unclosed shifts left in lastShift.
    for hid in lastShift.keys():
        # Close straggler shift.
        stime = lastShift[hid]['itime']
        etime = lastShift[hid]['otime']
        # Night shifts either span midnight, or begin very early
        # in the morning (before 5am) and ends by 8am (the latter
        # is quite arbitrary).
        if (stime.day != etime.day) or (stime.hour >= 18 and etime.hour < 24) or (stime.hour < 5 and etime.hour < 8):
            lastShift[hid]['shift'] = 'night'
        else:
            lastShift[hid]['shift'] = 'day'

        # Calculate average visit duration (overwrites cumulative visit duration stored in avgvdur)
        lastShift[hid]['avgvdur'] = lastShift[hid]['avgvdur']/(60*lastShift[hid]['vcount'])
        # Calculate number of unique rooms (overwrites set of rids)
        lastShift[hid]['rcount'] = len(lastShift[hid]['rcount'])

        
    # Append all of the newly closed stragglers to list of shifts.
    for hcw in lastShift.keys():
        if lastShift[hcw]['duration']/60 < (lastShift[hcw]['avgvdur'] * lastShift[hcw]['vcount']):
            print("Warning: shift duration less than average visit duration for shift {}.".format(lastShift[hcw]['shid']))
            import pdb; pdb.set_trace()
            
    newShifts.extend(lastShift.values())

    # Sort shifts by itime and jtid then print them out; would have
    # been more efficient to maintain in sorted order all along.
    newShifts.sort(key=lambda x: (x['itime'], x['jtid']))

    # All done.
    print("Imputed {} shifts between {} and {}.".format(len(newShifts), visits[0]['itime'], visits[-1]['otime']))

    return newShifts

def shift_statistics(newShifts):
    # Fourth, output visit/shift statistics to file.dat.
    jtypes = {0:('Unknown', 'NA'), 1:('Nursing', 'Nur'), 2:('Physical Therapy', 'PT'), 3:('Respiratory Therapy','RT'), 
              4:('Social Work', 'SW'), 5:('Occupational Therapy', 'OT'), 6:('Physician','MD'), 7:('Nursing Assistant','NAsst'),
              8:('Case Manager','CM'), 9:('Transporter','Tr'), 10:('Housekeeping','HK'), 11:('Phlebotomist','Phlb'),
              12:('Administration','Adm'), 13:('Food Service','Food'), 14:('Mid-level Provider','MLP'), 
              15:('Speech Therapy','SpT'), 16:('Quality Staff','Qual'), 17:('Clerical','Clk'), 18:('Pharmacy','Pharm'),
              19:('Spiritual Care/Chaplain', 'Clgy',), 20:('Laboratory Staff','Lab'), 21:('Nursing Management','NMgmt'),
              22:('Surgical Technicians','SrgT'), 23:('Dietitian','Diet'), 24:('Mental Health','Psych'), 
              25:('Paramedics','EMT'), 26:('Nurse Educators','NEd'), 27:('InfoTech','IT'), 28:('Radiology Tech','RadT'), 
              29:('Volunteer/Interpreter','Vol'), 30:('Resident Physicians','ResMD'),
              31:('Patient Representative','PRep'), 32:('Facilities/Maintenance','FM')}

    # visit count by shift by jtid
    visit_counts_by_jtid = { j:[ shift['vcount'] for shift in newShifts if shift['jtid'] == j ] for j in jtypes.keys() }
    # shift count by jtid
    shift_count_by_jtid = { j:len(visit_counts_by_jtid[j]) for j in visit_counts_by_jtid.keys() }

    # Shift efficiency and shift uniformity; remember, shift durations are hours while visit durations are minutes
    shift_efficiency = { j:sum([ (shift['avgvdur']*shift['vcount'])/(int(shift['duration'])/60) 
                                for shift in newShifts if shift['jtid'] == j ])/shift_count_by_jtid[j] 
                                for j in shift_count_by_jtid.keys() if shift_count_by_jtid[j] > 0 }

    shift_uniformity = { j:sum([ (shift['rcount']/shift['vcount']) 
                                for shift in newShifts if shift['jtid'] == j ])/shift_count_by_jtid[j] 
                                for j in shift_count_by_jtid.keys() if shift_count_by_jtid[j] > 0 }
    
    return {
        'shift_stats': {
            'shift_efficiency': shift_efficiency,
            'shift_uniformity': shift_uniformity
        }
    }



def create_visits_given_edgelist_default_duration(utid, edge_list, inv_maps=None, max_hid_idx=None):
    '''
    Given an edgelist, return the visit dictionary
    {'utid': 1, 'hid': 1, 'rid': 1, 'itime': '2023-01-01 08:00:00', 'duration': 120},
    '''
    visits = []
    default_duration = 120
    
    for src, dst, t in edge_list:
        itime = datetime.datetime.strptime(START_ITIME, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(seconds=t)
        itime = itime.timestamp()
        if inv_maps:
            # if the src/dst is greater than max_hid_idx, then it is a rid, otherwise it is a hid
            if src > max_hid_idx:
                rid = inv_maps['rid'][src]
                if dst > max_hid_idx:
                    raise ValueError('Error: both src and dst are rid')
                else:
                    hid = inv_maps['hid'][dst]
            elif dst > max_hid_idx:
                rid = inv_maps['rid'][dst]
                if src > max_hid_idx:
                    raise ValueError('Error: both src and dst are rid')
                else:
                    hid = inv_maps['hid'][src]
            else:
                raise ValueError('Error: both src and dst are hid')
        else:
            hid = src
            rid = dst
            
        visits.append({'utid': utid, 'hid': hid, 'rid': rid, 'itime': itime, 'duration': default_duration})
        
    return visits

def create_visits_given_edgelist(utid, edge_list, inv_maps=None, max_hid_idx=None):
    '''
    
    '''
    visits = []
    start_itime = datetime.datetime.strptime(START_ITIME, '%Y-%m-%d %H:%M:%S')
    
    for src, dst, t, d in edge_list:
        itime = start_itime + datetime.timedelta(seconds=t)
        itime = itime.timestamp()
        if inv_maps:
            # if the src/dst is greater than max_hid_idx, then it is a rid, otherwise it is a hid
            if src > max_hid_idx:
                rid = inv_maps['rid'][src]
                if dst > max_hid_idx:
                    raise ValueError('Error: both src and dst are rid, src: {}, dst: {}, max_hid_idx: {}'.format(src, dst, max_hid_idx))
                else:
                    hid = inv_maps['hid'][dst]
            elif dst > max_hid_idx:
                rid = inv_maps['rid'][dst]
                if src > max_hid_idx:
                    raise ValueError('Error: both src and dst are rid, src: {}, dst: {}, max_hid_idx: {}'.format(src, dst, max_hid_idx))
                else:
                    hid = inv_maps['hid'][src]
            else:
                raise ValueError('Error: both src and dst are hid, src: {}, dst: {}, max_hid_idx: {}'.format(src, dst, max_hid_idx))
        else:
            hid = src
            rid = dst
            
        visits.append({'utid': utid, 'hid': hid, 'rid': rid, 
                       'itime': itime, 'duration': d})
        
    return visits



def get_shift_stats(utid, edge_list, inv_maps, hid2jtid_map, cfg, max_hid_idx):
    if len(edge_list[0]) == 3: # no duration information
        visits = create_visits_given_edgelist_default_duration(utid, edge_list, inv_maps=inv_maps, max_hid_idx=max_hid_idx)
    elif len(edge_list[0]) == 4: # with duration information
        visits = create_visits_given_edgelist(utid, edge_list, inv_maps=inv_maps, max_hid_idx=max_hid_idx)
        
    newShifts = shift_construction(cfg, visits, hid2jtid_map=hid2jtid_map)
    shift_hr_stats = shift_statistics(newShifts)
    return shift_hr_stats

def convert_list_int64_to_int(d):
    for k, v in d.items():
        if isinstance(v, list):
            d[k] = [int(x) for x in v]
        elif isinstance(v, dict):
            d[k] = convert_list_int64_to_int(v)
    return d

def count_invalid_edges(edge_list, max_hid_idx):
    invalid_edges = 0
    for src, dst, t, d in edge_list:
        if src > max_hid_idx and dst > max_hid_idx:
            invalid_edges += 1
        
    return invalid_edges

def shift_uniformity_efficiency_by_jtid(utids, all_shift_stats, save_dir):
    avg_utid_diff_efficiency = {}
    avg_utid_diff_uniformity = {}
    
    for utid in utids:
        efficiency_data = []
        uniformity_data = []
        all_jtids = set()
        
        # First, collect all unique jtids across all methods
        for name in all_shift_stats[utid]:
            all_shift_stats[utid][name] = convert_list_int64_to_int(all_shift_stats[utid][name])
            shift_efficiency = all_shift_stats[utid][name]['shift_stats']['shift_efficiency']
            for jtid in shift_efficiency:
                all_jtids.add(jtid)
        
        all_jtids = sorted(all_jtids)
        
        # Then collect data, handling missing jtids
        for name in all_shift_stats[utid]:
            shift_efficiency = all_shift_stats[utid][name]['shift_stats']['shift_efficiency']
            shift_uniformity = all_shift_stats[utid][name]['shift_stats']['shift_uniformity']
            shift_efficiency = {k: round(v, 4) for k, v in shift_efficiency.items()}
            shift_uniformity = {k: round(v, 4) for k, v in shift_uniformity.items()}
            
            # Add data for all jtids, even if missing
            for jtid in all_jtids:
                efficiency_value = shift_efficiency.get(jtid, float('nan'))
                uniformity_value = shift_uniformity.get(jtid, float('nan'))
                
                efficiency_data.append({'jtid': jtid, 'name': name, 'value': efficiency_value})
                uniformity_data.append({'jtid': jtid, 'name': name, 'value': uniformity_value})
        
        # Plot efficiency
        efficiency_df = pd.DataFrame(efficiency_data)
        efficiency_df_pivot = efficiency_df.pivot(index='jtid', columns='name', values='value')
        efficiency_df_pivot.to_csv(f'{save_dir}/utid_{utid}_shift_efficiency.csv')
        
        # Compute the average absolute difference between available methods with original data for each jtid
        avg_diff_efficiency = {}
        for method in all_shift_stats[utid]:
            if method == 'ogdata': continue
            diff = efficiency_df_pivot['ogdata'] - efficiency_df_pivot[method]
            diff = diff.fillna(1)  # Fill missing values with 1 - pentalty for missing/redudant jtids
            avg_diff_efficiency[method] = np.mean(np.abs(diff))
        # 
        avg_utid_diff_efficiency[utid] = avg_diff_efficiency
        # round values
        avgdiff_plot = {k: round(v, 4) for k, v in avg_diff_efficiency.items()}
        plt.figure(figsize=(20, 6))
        ax = sns.barplot(x='jtid', y='value', hue='name', data=efficiency_df)
        plt.title(f'Shift Efficiency by Jtid for Utid {utid} - Avg Diff: {avgdiff_plot}')
        plt.xlabel('Jtid')
        plt.ylabel('Shift Efficiency')
        plt.legend(title='Method')
        
        # Add text annotations
        for p in ax.patches:
            if p.get_height() < 1e-5: continue 
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 9), textcoords='offset points', rotation=90, fontsize=8)
            
        figsavepath=f'{save_dir}/utid_{utid}_shift_efficiency.png'
        plt.tight_layout()
        plt.savefig(figsavepath)
        print(f'Saved {figsavepath}')
        plt.close()
        
        # Plot uniformity
        uniformity_df = pd.DataFrame(uniformity_data)
        uniformity_df_pivot = uniformity_df.pivot(index='jtid', columns='name', values='value')
        uniformity_df_pivot.to_csv(f'{save_dir}/utid_{utid}_shift_uniformity.csv')
        avg_diff_uniformity = {}
        for method in all_shift_stats[utid]:
            if method == 'ogdata':
                continue
            diff = uniformity_df_pivot['ogdata'] - uniformity_df_pivot[method]
            diff = diff.fillna(1)
            avg_diff_uniformity[method] = np.mean(np.abs(diff))
        #
        avg_utid_diff_uniformity[utid] = avg_diff_uniformity
        # round values
        avgdiff_plot = {k: round(v, 4) for k, v in avg_diff_uniformity.items()}
        plt.figure(figsize=(20, 6))
        ax = sns.barplot(x='jtid', y='value', hue='name', data=uniformity_df)
        plt.title(f'Shift Uniformity by Jtid for Utid {utid} - Avg Diff: {avgdiff_plot}')
        plt.xlabel('Jtid')
        plt.ylabel('Shift Uniformity')
        plt.legend(title='Method')
        
        # Add text annotations
        for p in ax.patches:
            if p.get_height() < 1e-5: continue 
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 9), textcoords='offset points', rotation=90, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/utid_{utid}_shift_uniformity.png')
        plt.close()
        
    return avg_utid_diff_efficiency, avg_utid_diff_uniformity

def plot_duration_histograms(data, name, save_dir, picked_hids):
    """
    Plot histograms with KDE of duration values for each healthcare worker.
    
    Args:
        data: DataFrame containing the data
        name: String identifier for the dataset (e.g., 'ogdata', 'bittigger')
        save_dir: Directory to save the plots
    """
    # Group data by healthcare worker (HID)
    duration_dir = Path(save_dir) / 'duration_histograms' / name
    duration_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot overall duration distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data['duration'], kde=True, bins=50)
    plt.title(f'Overall Duration Distribution - {name}')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Count')
    plt.savefig(f'{duration_dir}/overall_duration.png')
    plt.close()
    
    # Group data by healthcare worker (HID/u)
    grouped_by_hid = data.groupby('u')
    
    # Plot individual histograms for top HCWs
    for hid in picked_hids:
        group = grouped_by_hid.get_group(hid)
        if len(group) < 10:  # Skip if not enough data points
            continue
        
        plt.figure(figsize=(10, 6))
        sns.histplot(group['duration'], kde=True, bins=30)
        plt.title(f'Duration Distribution for HCW {hid} - {name}')
        plt.xlabel('Duration (seconds)')
        plt.ylabel('Count')
        plt.savefig(f'{duration_dir}/hcw_{hid}_duration.png')
        plt.close()
    
    print(f"Saved duration histograms for {name} to {duration_dir}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate shifts from generated temporal graphs.')
    parser.add_argument('--opath', type=str, 
                        default='DATA/processed_data/shift1/all_unittypes_refined.csv',
                        help='Path to original data CSV file')
    parser.add_argument('--bpaths', type=str, nargs='+', 
                        default=['Bipartite_TIGGER/results/shift1_addheads_fixlb/postpro_best/sampled_graph_0.csv'],
                        help='List of paths to generated data CSV files')
    parser.add_argument('--bnames', type=str, nargs='+', 
                        default=['bittigger'],
                        help='List of names corresponding to the generated data CSV files')
    parser.add_argument('--data_dir', type=str,
                        default='DATA/processed_data/shift1',
                        help='Directory containing processed data files')
    parser.add_argument('--hid_to_jtid_path', type=str, 
                        default='DATA/hid_to_jtid_mapping.json',
                        help='Path to JSON file mapping hid to jtid')
    parser.add_argument('--save_dir', type=str, 
                        default='shift_eval_plots',
                        help='Directory to save evaluation plots')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    save_dir = args.save_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Load inverse mapping
    datadir = args.data_dir
    inv_maps = pickle.load(open(f'{datadir}/all_unittypes_refined_inv_map.pkl', 'rb'))
    idx_to_processed_hid_map = inv_maps['u'] # idx : processed_hid
    idx_to_processed_rid_map = inv_maps['i']

    max_hid_idx = max(idx_to_processed_hid_map.keys())
    inv_maps = {'hid': idx_to_processed_hid_map, 'rid': idx_to_processed_rid_map}

    with open(args.hid_to_jtid_path, 'r') as f:
        hid2jtid_map = json.load(f)
        # convert string key to int key
        hid2jtid_map = {int(k): v for k, v in hid2jtid_map.items()}

    cfg = {
        'shgap': 60 * 60 * 10,  # 10 hour gap
        'stime': '2018-09-15',
        'etime': '2019-03-15'
    }

    all_shift_stats = defaultdict(dict)
    num_invalid_edges = defaultdict(int)
    utids_16 = [v for v in map(int, "0 2 3 4 5 8 9 12 13 14 19 21 24 26 28 30".split())]
    utids_10 = [v for v in map(int, "0 2 3 5 12 14 21 24 26 28 ".split())]
    utids_2 = [v for v in map(int, "0 2".split())]
    utids = utids_10
    
    # odata_path = 'DATA/processed_data/shift1/all_unittypes_refined.csv'
    # bittigger_path = 'Bipartite_TIGGER/results/shift1_addheads_fixlb/postpro_best/sampled_graph_0.csv' 
    
    odata_path = args.opath
    bpaths = args.bpaths
    bnames = args.bnames
    all_names = ['ogdata'] + bnames
    all_paths = [odata_path] + bpaths

    num_invalid_edges = defaultdict(int)

    # 
    picked_hids = None 
    
    for name, fpath in zip(all_names, all_paths):
        print(f'Processing {fpath}')
        save_path = fpath.replace('.csv', '_shift_hr_stats.pkl')
        assert save_path != fpath
        
        data = pd.read_csv(fpath) # u,i,ts,label,otime,duration,uid,shid,u_budget,i_budget
        data = data[['u', 'i', 'ts', 'label', 'duration']]
        
        if name == 'ogdata':
            # pick top hids with most visits
            top_hids = data['u'].value_counts().head(10).index.tolist()
            picked_hids = top_hids
            
        # Plot duration histograms
        plot_duration_histograms(data, name, save_dir, picked_hids)
            
        for utid in utids:
            visits = data[data['label'] == utid]        
            # assert tempG_data.t is sorted in ascending order
            if not np.all(visits.ts[1:].values >= visits.ts[:-1].values):
                print(f'Utid {utid} - Not sorted for {name}')
                # concat two rows at violation
                idx = np.where(visits.ts.values[1:] < visits.ts.values[:-1])[0]
                print(visits.iloc[idx])
                print(visits.iloc[idx+1])
                import pdb; pdb.set_trace()

            # assert the next visit itime is greater than the current visit otime 
            hcw_visits = visits.groupby('u')
            for hid, group in hcw_visits:
                assert np.all(group.ts[1:].values >= group.ts[:-1].values)
                # assert np.all(group.ts.values[1:] >= group.ts.values[:-1] + group.duration.values[:-1])
                # find where it violates the above condition
                idx = np.where(group.ts.values[1:] < group.ts.values[:-1] + group.duration.values[:-1])[0]
                if len(idx) > 0:
                    print(f'Utid {utid} - Violation in hid {hid} for {name}')
                    print(group.iloc[idx])
                    print(group.iloc[idx+1])
                    import pdb; pdb.set_trace()
                
            edge_list = list(zip(visits.u, visits.i, visits.ts, visits.duration))
            num_invalid_edges[utid] = count_invalid_edges(edge_list, max_hid_idx)
            print(f'Utid {utid} - Invalid edges for {name}: {num_invalid_edges[utid]}')
            
            shift_hr_stats = get_shift_stats(utid, edge_list, inv_maps, hid2jtid_map, cfg, max_hid_idx)

            all_shift_stats[utid][name] = shift_hr_stats

        # save all_shift_stats
        with open(save_path, 'wb') as f:
            pickle.dump(all_shift_stats, f)
        print(f'Saved {save_path}')
        print()

    # shift efficiency and uniformity by jtid for each utid
    avg_utid_diff_efficiency, avg_utid_diff_uniformity = \
        shift_uniformity_efficiency_by_jtid(utids, all_shift_stats, save_dir)

    # compute average efficiency and uniformity difference over all utids
    avg_diff_efficiency = {name: np.mean([v[name] for v in avg_utid_diff_efficiency.values()]) for name in bnames}
    avg_utid_diff_efficiency['avg'] = avg_diff_efficiency
    
    avg_diff_uniformity = {name: np.mean([v[name] for v in avg_utid_diff_uniformity.values()]) for name in bnames}
    avg_utid_diff_uniformity['avg'] = avg_diff_uniformity
    
    print(f'Average Efficiency Difference: {avg_diff_efficiency}')
    print(f'Average Uniformity Difference: {avg_diff_uniformity}')
    
    # save avg_utid_diff_efficiency and avg_utid_diff_uniformity in json format
    with open(f'{save_dir}/avg_utid_diff_efficiency.json', 'w') as f:
        json.dump(avg_utid_diff_efficiency, f, indent=4)
    print(f'Saved {save_dir}/avg_utid_diff_efficiency.json')
    with open(f'{save_dir}/avg_utid_diff_uniformity.json', 'w') as f:
        json.dump(avg_utid_diff_uniformity, f, indent=4)
    print(f'Saved {save_dir}/avg_utid_diff_uniformity.json')