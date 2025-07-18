import json
import wandb
import os
import argparse
import numpy as np
import glob

# print as row table
def print_metrics(methods_metric_values, std_over_utids_metrics, metrics_to_log):
    print(f"{'Method':<30}", end="\t")
    for metric in metrics_to_log:
        print(f"{metric:<16}", end="\t")
    print()
    # print actual median
    # for method in methods_metric_values.keys():
    #     print(f"{f'Actual_Median_({method})':<30}", end="\t")
    #     for metric in metrics_to_log:
    #         value = methods_metric_values[method]['actual_median'].get(metric, 'N/A')
    #         if isinstance(value, float):
    #             value = f"{value:.6f}"
    #         print(f"{value:<15}", end="\t")
    #     print()
    for method in methods_metric_values.keys():
        print(f"{method:<30}", end="\t")
        for metric in metrics_to_log:
            value = methods_metric_values[method]['median_abs_error'].get(metric, 'N/A')
            std = std_over_utids_metrics[method]['median_abs_error'].get(metric, 'N/A')
            if isinstance(value, float):
                value = f"{value:.4f}"
            if isinstance(std, float):
                std = f"{std:.4f}"
            print(f"{value:<7} ± {std:<6}", end="\t")
        print()

def print_metrics_as_latex(methods_metric_values, std_over_utids_metrics, metrics_to_log):
    print(f"{'Method':<30}", end=" & ")
    for metric in metrics_to_log:
        print(f"{metric:<45}", end=" & ")
    print("\\\\")
    best_method = {}
    best_values = {metric: float('inf') for metric in metrics_to_log}
    for metric in metrics_to_log:
        for method, metrics in methods_metric_values.items():
            value = metrics['median_abs_error'][metric]
            if value < best_values[metric]:
                best_values[metric] = value
                best_method[metric] = method

    for method in methods_metric_values.keys():
        method_disp = method.replace('_', '\_')
        # Mean row
        print(f"{method_disp:<30}", end=" & ")
        for metric in metrics_to_log:
            value = methods_metric_values[method]['median_abs_error'].get(metric, 'N/A')
            if isinstance(value, float):
                value_str = f"{value:.4f}"
            else:
                value_str = str(value)
            if method == best_method[metric]:
                print(f"\\textbf{{{value_str}}}", end=" & ")
            else:
                print(f"{value_str}", end=" & ")
        print("\\\\")
        # Std row
        print(f"{'':<30}", end=" & ")
        for metric in metrics_to_log:
            std = std_over_utids_metrics[method]['median_abs_error'].get(metric, 'N/A')
            if isinstance(std, float):
                std_str = f"{std:.4f}"
            else:
                std_str = str(std)
            if method == best_method[metric]:
                print(f"(\\textbf{{{std_str}}})", end=" & ")
            else:
                print(f"({std_str})", end=" & ")
        print("\\\\ \\hline")

    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log grouped bar chart of average metrics to wandb")
    parser.add_argument("--method_names", nargs='+', type=str, help="List of method names")
    parser.add_argument("--save_dirs", nargs='+', type=str, help="List of directories to save the results")
    parser.add_argument("--utids", nargs='+', type=float, help="List of utids of the generated graphs")
    parser.add_argument("--data_name", type=str, help="Name of the dataset")
    args = parser.parse_args()
    
    assert len(args.save_dirs) == len(args.method_names)
    utids = args.utids
    
    metrics_all = {}
    
    for utid in utids:
        metrics_all[utid] = {}
        for result_dir, method_name in zip(args.save_dirs, args.method_names):
            all_samples_metrics = {}
            # tigger-alike
            try:
                fnames = glob.glob(os.path.join(result_dir, f"sampled_graph_*_utid_{int(utid)}_metrics.json"))
                for fname in fnames:
                    print("reading ", fname)
                    sample_idx = int(fname.split("_")[-4])
                    with open(fname, 'r') as f:
                        metrics = json.load(f)
                    all_samples_metrics[sample_idx] = metrics
            except:
                pass
            
            if len(all_samples_metrics) == 0:
                print(f"No metrics found for {method_name} with utid {utid}")
                continue
            
            metrics_all[utid][method_name] = all_samples_metrics

      
    # compute average metrics over utids
    metric_groups = ['median_abs_error', 'mean_abs_error', 'actual_median']
    metrics_to_log = ['%_edge_overlap', 'd_mean', 'wedge_count', 'power_law_exp',
                    'rel_edge_distr_entropy', 'LCC', 'n_components', 
                    'betweenness_centrality_mean', 'closeness_centrality_mean']

    # given metrics_all[utid][method_name][sample_idx][metric_group][metric]
    # average over utids
    avg_over_utids_metrics = {}
    std_over_utids_metrics = {}
    for method_name in metrics_all[utids[0]].keys():
        avg_over_utids_metrics[method_name] = {}
        std_over_utids_metrics[method_name] = {}
        for sample_idx in metrics_all[utids[0]][method_name].keys():
            avg_over_utids_metrics[method_name][sample_idx] = {}
            std_over_utids_metrics[method_name][sample_idx] = {}
            
            for metric_group in metric_groups:
                if metric_group not in avg_over_utids_metrics[method_name]:
                    avg_over_utids_metrics[method_name][sample_idx][metric_group] = {}
                    std_over_utids_metrics[method_name][sample_idx][metric_group] = {}
                for metric in metrics_to_log:
                    values = []
                    for utid in metrics_all.keys():
                        if metric == '%_edge_overlap' and metric_group == 'actual_median':
                            values.append(100.0)  # for actual median, we assume 100% edge overlap
                        else:
                            try:
                                values.append(metrics_all[utid][method_name][sample_idx][metric_group][metric])
                            except KeyError:
                                raise KeyError(f"KeyError for {utid}, {method_name}, {sample_idx}, {metric_group}, {metric}")
                            
                    avg_over_utids_metrics[method_name][sample_idx][metric_group] [metric] = np.mean(values)
                    std_over_utids_metrics[method_name][sample_idx][metric_group][metric] = np.std(values)
 
    # print(f"\nAverage Metrics over UTIDs:")
    # print_metrics(avg_over_utids_metrics, std_over_utids_metrics, metrics_to_log)
    
    # give avg_over_utids_metrics
    avg_over_samples_metrics = {}
    std_over_samples_metrics = {}
    for method_name in metrics_all[utids[0]].keys():
        avg_over_samples_metrics[method_name] = {}
        std_over_samples_metrics[method_name] = {}
        for metric_group in metric_groups:
            avg_over_samples_metrics[method_name][metric_group] = {}
            std_over_samples_metrics[method_name][metric_group] = {}
            for metric in metrics_to_log:
                values = []
                for sample_idx in avg_over_utids_metrics[method_name].keys():
                    try:
                        values.append(
                            avg_over_utids_metrics[method_name][sample_idx][metric_group][metric]
                        )
                    except KeyError:
                        raise KeyError(f"KeyError for {method_name}, {sample_idx}, {metric_group}, {metric}")
                
                avg_over_samples_metrics[method_name][metric_group][metric] = np.mean(values)
                std_over_samples_metrics[method_name][metric_group][metric] = np.std(values)
    
    picked_method = args.method_names[0]
    avg_over_samples_metrics['Actual_Median'] = {mg: {} for mg in metric_groups}
    std_over_samples_metrics['Actual_Median'] = {mg: {} for mg in metric_groups}
    avg_over_samples_metrics['Actual_Median']['median_abs_error'] = {}
    avg_over_samples_metrics['Actual_Median']['mean_abs_error'] = {}
    std_over_samples_metrics['Actual_Median']['median_abs_error'] = {}
    std_over_samples_metrics['Actual_Median']['mean_abs_error'] = {}
    
    for metric in metrics_to_log:
        values = []
        for sample_idx in avg_over_utids_metrics[method_name].keys():
            try:
                values.append(
                    avg_over_utids_metrics[picked_method][sample_idx]['actual_median'][metric]
                )
            except KeyError:
                raise KeyError(f"KeyError for {picked_method}, {sample_idx}, 'actual_median', {metric}")
            
        values = np.array(values)
        avg_over_samples_metrics['Actual_Median']['median_abs_error'][metric] = np.mean(values)
        avg_over_samples_metrics['Actual_Median']['mean_abs_error'][metric] = np.mean(values)
        std_over_samples_metrics['Actual_Median']['median_abs_error'][metric] = np.std(values)
        std_over_samples_metrics['Actual_Median']['mean_abs_error'][metric] = np.std(values)
    #   
    # compute average metrics over sampled graphs
    print(f"\nGenerated Graph Metrics:")
    print_metrics(avg_over_samples_metrics, std_over_samples_metrics, metrics_to_log)
    print("\n\n")
    print_metrics_as_latex(avg_over_samples_metrics, std_over_samples_metrics, metrics_to_log)
