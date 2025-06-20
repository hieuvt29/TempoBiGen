# adding duration for generated temporal graph
import os
import random
import pandas as pd
import datetime
from collections import defaultdict,Counter
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
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sklearn
from sklearn.model_selection import train_test_split
import csv
import h5py
from torch.autograd import Variable
from torch.nn import functional as F
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from model_classes.transductive_model import CondBipartiteEventLSTM
from torch.utils.data import DataLoader, TensorDataset

from metrics.metric_utils import (
    get_numpy_matrix_from_adjacency,
    get_adj_graph_from_random_walks,
    get_total_nodes_and_edges_from_temporal_adj_list_in_time_range,
    get_adj_origina_graph_from_original_temporal_graph,
)
from metrics.metric_utils import (
    sample_adj_graph_multinomial_k_inductive,
    sample_adj_graph_topk,
)
import sklearn
from collections import OrderedDict

from tgg_utils import *
from train_transductive import seed_everything, config_logging
import glob 
from pathlib import Path
import logging 
import random
import json

EPS = 1e-6 

from scipy.stats import gamma, expon, lognorm, weibull_min, truncnorm, norm
from scipy.special import gamma as gamma_func  # Import gamma function specifically
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

# import warnings
# warnings.filterwarnings("error")

np.int = int
np.float = float
PROJECT_NAME = "ECMLPKDD25_TIGGER"
USE_WANDB = True
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



def fit_lognormal_mixture(data, n_components=2, max_iter=1000):
    """Fit mixture of Log-normal distributions using EM algorithm"""
    # Filter out non-positive values (log is undefined for <= 0)
    data = np.array(data)
    valid_idx = data > 0
    
    if np.sum(valid_idx) < len(data):
        logging.info(f"Warning: Removed {len(data) - np.sum(valid_idx)} non-positive values for log-normal fitting")
    
    if np.sum(valid_idx) < 10:  # If too few valid data points
        # Return default parameters
        return np.ones(n_components)/n_components, np.ones(n_components), np.ones(n_components)
    
    data = data[valid_idx]
    N = len(data)
    
    # Initialize parameters
    weights = np.ones(n_components) / n_components
    # means = np.random.uniform(0, 2, n_components)  # Log-space means
    # sigmas = np.random.uniform(0.1, 1, n_components)  # Log-space standard deviations
    
    # set initial means, sigmas based on data
    mean = np.mean(np.log(data))
    means = np.random.normal(mean.item(), 2, n_components)
    
    sigma = np.std(np.log(data))
    sigmas = np.random.normal(sigma.item(), 1, n_components)
    
    # Pre-compute log data to avoid recomputing
    log_data = np.log(data + 1e-10)  # Small epsilon for stability
    
    for _ in range(max_iter):
        # save old parameters for early stopping
        old_means = np.copy(means)
        old_sigmas = np.copy(sigmas)
        
        # E-step: compute responsibilities
        resp = np.zeros((N, n_components))
        for k in range(n_components):
            # Use lognorm.pdf with safeguards
            try:
                resp[:, k] = weights[k] * lognorm.pdf(data, s=max(sigmas[k], 1e-6), scale=np.exp(means[k]))
            except Warning:
                resp[:, k] = weights[k] * 1e-10  # Small default if computation fails
        
        # Handle numerical instability in normalization
        resp_sum = resp.sum(axis=1, keepdims=True)
        resp_sum[resp_sum < 1e-10] = 1e-10  # Avoid division by zero
        resp /= resp_sum
        
        # M-step: update parameters
        Nk = resp.sum(axis=0)
        Nk[Nk < 1e-10] = 1e-10  # Avoid division by zero
        weights = Nk / N
        
        for k in range(n_components):
            # MLE for log-normal: work in log space with numerical safeguards
            weighted_sum = np.sum(resp[:, k] * log_data)
            means[k] = weighted_sum / Nk[k]
            
            # Compute variance with stability checks
            var_k = np.sum(resp[:, k] * (log_data - means[k])**2) / Nk[k]
            
            # Ensure variance is positive and not too small
            if np.isnan(var_k) or var_k < 1e-6:
                var_k = 1e-6
                
            sigmas[k] = np.sqrt(var_k)
            
        # early stopping when the change in parameters is small
        if np.all(np.abs(means - old_means) < 1e-6) and np.all(np.abs(sigmas - old_sigmas) < 1e-6):
            break
    
    # Final validation of parameters
    for k in range(n_components):
        if np.isnan(means[k]) or np.isinf(means[k]):
            means[k] = 0.0
        if np.isnan(sigmas[k]) or np.isinf(sigmas[k]) or sigmas[k] < 1e-6:
            sigmas[k] = 1.0
    
    return weights, means, sigmas

def plot_lognormal_mixture(ax, data, weights, means, sigmas, color='green', show_samples=False):
    """Plot fitted Log-normal mixture components and their sum"""
    x = np.linspace(min(data), max(data), 1000)
    total_pdf = np.zeros_like(x)
    
    for i in range(len(weights)):
        component_pdf = weights[i] * lognorm.pdf(x, s=sigmas[i], scale=np.exp(means[i]))
        total_pdf += component_pdf
        # Plot individual components
        ax.plot(x, component_pdf, '-', linewidth=0.8, color=color, alpha=0.3)
    
    # Plot total mixture
    ax.plot(x, total_pdf, color=color, label='Log-normal Mixture', linewidth=2)
    
    if show_samples:
        # Sample from the mixture and plot histogram
        samples = np.concatenate([np.random.lognormal(means[i], sigmas[i], int(weights[i] * len(data))) for i in range(len(weights))])
        ax.hist(samples, bins=50, density=True, color=color, alpha=0.3, label='Samples')
    
    # Calculate log likelihood and number of non-neglected components
    log_likelihood = np.sum(np.log(np.sum([weights[i] * lognorm.pdf(data, s=sigmas[i], scale=np.exp(means[i])) for i in range(len(weights))], axis=0)))
    non_neglected_components = np.sum(weights > 1e-3)
    ax.set_xlabel(f'Log Likelihood: {log_likelihood:.2f}, Non-neglected Components: {non_neglected_components}')
    
    return total_pdf

def fit_dpm_lognormal_mixture(data, max_components=10, alpha=0.5, max_iter=100):
    """Fit Dirichlet Process Mixture of Log-normal distributions using variational inference"""
    # Filter out non-positive values (log is undefined for <= 0)
    data = np.array(data)
    valid_idx = data > 0
    
    if np.sum(valid_idx) < len(data):
        logging.info(f"Warning: Removed {len(data) - np.sum(valid_idx)} non-positive values for DPM log-normal fitting")
    
    if np.sum(valid_idx) < 10:  # If too few valid data points
        # Return default parameters
        return np.array([1.0]), np.array([[0.0, 1.0]])
    
    data = data[valid_idx]
    N = len(data)
    log_data = np.log(data)
    
    # Initialize with K-means
    K = min(max_components, N // 10 + 1)  # Ensure we have enough data points per component
    kmeans = KMeans(n_clusters=K, random_state=42).fit(log_data.reshape(-1, 1))
    
    # Initialize mixture parameters
    cluster_assignments = kmeans.labels_
    unique_clusters = np.unique(cluster_assignments)
    K_active = len(unique_clusters)
    
    # Initialize component parameters (mean, precision)
    components = []
    weights = np.zeros(K_active)
    
    for k, cluster_id in enumerate(unique_clusters):
        cluster_data = log_data[cluster_assignments == cluster_id]
        mean = np.mean(cluster_data)
        var = np.var(cluster_data)
        if var < 1e-6:
            var = 1e-6
        precision = 1.0 / var
        components.append([mean, precision])
        weights[k] = len(cluster_data) / N
    
    # Convert to numpy array
    components = np.array(components)
    
    # Variational inference for DP mixture
    for _ in range(max_iter):
        # E-step: compute responsibilities
        log_resp = np.zeros((N, K_active))
        
        for k in range(K_active):
            mean, precision = components[k]
            variance = 1.0 / precision
            log_resp[:, k] = np.log(weights[k] + 1e-10) - 0.5 * np.log(2 * np.pi * variance) - \
                            0.5 * precision * (log_data - mean)**2
        
        # Normalize responsibilities
        log_resp_max = np.max(log_resp, axis=1, keepdims=True)
        resp = np.exp(log_resp - log_resp_max)
        resp_sum = resp.sum(axis=1, keepdims=True)
        resp_sum[resp_sum < 1e-10] = 1e-10
        resp = resp / resp_sum
        
        # M-step
        Nk = resp.sum(axis=0)
        active_components = Nk > (alpha / K_active)
        
        if np.sum(active_components) == 0:
            # If no active components, keep the strongest one
            active_components[np.argmax(Nk)] = True
        
        # Update weights with Dirichlet prior
        weights = (Nk + alpha/K_active) / (N + alpha)
        
        # Update components for active clusters
        new_components = []
        new_weights = []
        
        for k in range(K_active):
            if active_components[k]:
                # Update mean and precision for active component
                resp_k = resp[:, k]
                weighted_sum = np.sum(resp_k * log_data)
                weighted_mean = weighted_sum / (Nk[k] + 1e-10)
                
                weighted_var = np.sum(resp_k * (log_data - weighted_mean)**2) / (Nk[k] + 1e-10)
                if weighted_var < 1e-6:
                    weighted_var = 1e-6
                
                weighted_precision = 1.0 / weighted_var
                
                new_components.append([weighted_mean, weighted_precision])
                new_weights.append(weights[k])
        
        # Update active components
        components = np.array(new_components)
        weights = np.array(new_weights)
        weights = weights / np.sum(weights)  # Renormalize weights
        K_active = len(components)
        
        # Check for convergence if K_active is stable
        if K_active <= 1:
            break
    
    # Return final parameters
    return weights, components

def sample_dpm_lognormal_mixture(weights, components, n_samples=1):
    """Sample from Dirichlet Process Mixture of Log-normal distributions"""
    samples = []
    for _ in range(n_samples):
        # Sample a component
        k = np.random.choice(len(weights), p=weights)
        mean, precision = components[k]
        std_dev = np.sqrt(1.0 / precision)
        
        # Sample from log-normal distribution
        sample = np.exp(np.random.normal(mean, std_dev))
        samples.append(sample)
    
    return np.array(samples)

def plot_dpm_lognormal_mixture(ax, data, weights, components, color='magenta', show_samples=False):
    """Plot fitted Dirichlet Process Mixture of Log-normal distributions"""
    x = np.linspace(min(data), max(data), 1000)
    total_pdf = np.zeros_like(x)
    
    for i in range(len(weights)):
        mean, precision = components[i]
        std_dev = np.sqrt(1.0 / precision)
        
        # Log-normal PDF using these parameters
        component_pdf = weights[i] * lognorm.pdf(x, s=std_dev, scale=np.exp(mean))
        total_pdf += component_pdf
        
        # Plot individual components
        ax.plot(x, component_pdf, '-', linewidth=0.8, color=color, alpha=0.3)
    
    # Plot total mixture
    ax.plot(x, total_pdf, color=color, label='DPM Log-normal', linewidth=2)
    
    if show_samples:
        # Sample from the mixture and plot histogram
        samples = sample_dpm_lognormal_mixture(weights, components, n_samples=len(data))
        ax.hist(samples, bins=50, density=True, color=color, alpha=0.3, label='Samples')
    
    # Calculate log likelihood and number of non-neglected components
    log_likelihood = np.sum(np.log(np.sum([weights[i] * lognorm.pdf(data, s=np.sqrt(1.0 / components[i][1]), scale=np.exp(components[i][0])) for i in range(len(weights))], axis=0)))
    non_neglected_components = np.sum(weights > 1e-10)
    ax.set_xlabel(f'Log Likelihood: {log_likelihood:.2f}, Non-neglected Components: {non_neglected_components}')
    
    return total_pdf

def fit_dpm_lognormal_mixture_pyro(data, max_components=10, alpha=0.5, max_iter=100, lr=0.05):
    """Fit Dirichlet Process Mixture of Log-normal distributions using Pyro"""
    # Filter out non-positive values (log is undefined for <= 0)
    data = np.array(data)
    valid_idx = data > 0
    
    if np.sum(valid_idx) < len(data):
        logging.info(f"Warning: Removed {len(data) - np.sum(valid_idx)} non-positive values for DPM log-normal fitting")
    
    if np.sum(valid_idx) < 10:  # If too few valid data points
        return np.array([1.0]), np.array([[0.0, 1.0]])
    
    data = data[valid_idx]
    N = len(data)
    log_data = np.log(data)
    
    # Convert data to torch tensor
    log_data = torch.tensor(log_data, dtype=torch.float32)
    
    def model(data):
        weights = pyro.sample("weights", dist.Dirichlet(torch.ones(max_components) * alpha / max_components))
        means = pyro.sample("means", dist.Normal(0.0, 10.0).expand([max_components]).to_event(1))
        stds = pyro.sample("stds", dist.HalfNormal(10.0).expand([max_components]).to_event(1))
        
        with pyro.plate("data", len(data)):
            assignment = pyro.sample("assignment", dist.Categorical(weights))
            pyro.sample("obs", dist.LogNormal(means[assignment], stds[assignment]), obs=data)
    
    def guide(data):
        weights_posterior = pyro.param("weights_posterior", torch.ones(max_components) / max_components, constraint=dist.constraints.simplex)
        means_posterior = pyro.param("means_posterior", torch.zeros(max_components))
        stds_posterior = pyro.param("stds_posterior", torch.ones(max_components), constraint=dist.constraints.positive)
        
        pyro.sample("weights", dist.Dirichlet(weights_posterior))
        pyro.sample("means", dist.Normal(means_posterior, 10.0).expand([max_components]).to_event(1))
        pyro.sample("stds", dist.HalfNormal(stds_posterior).expand([max_components]).to_event(1))
        
        with pyro.plate("data", len(data)):
            pyro.sample("assignment", dist.Categorical(weights_posterior))
    
    pyro.clear_param_store()
    svi = SVI(model, guide, Adam({"lr": lr}), loss=Trace_ELBO())
    
    for _ in range(max_iter):
        svi.step(log_data)
    
    weights = pyro.param("weights_posterior").detach().numpy()
    means = pyro.param("means_posterior").detach().numpy()
    stds = pyro.param("stds_posterior").detach().numpy()
    
    components = np.column_stack((means, 1.0 / (stds ** 2)))
    
    return weights, components

def fit_truncated_lognormal_mixture(data, upper_bounds, lower_bounds, n_components=2, max_iter=1000):
    """Fit mixture of truncated Log-normal distributions using EM algorithm"""
    # Filter out non-positive values (log is undefined for <= 0)
    data = np.array(data)
    upper_bounds = np.array(upper_bounds)
    lower_bounds = np.array(lower_bounds)
    valid_idx = data > 0
    
    if np.sum(valid_idx) < len(data):
        logging.info(f"Warning: Removed {len(data) - np.sum(valid_idx)} non-positive values for truncated log-normal fitting")
    
    if np.sum(valid_idx) < 10:  # If too few valid data points
        return np.ones(n_components)/n_components, np.ones(n_components), np.ones(n_components)
    
    data = data[valid_idx]
    upper_bounds = upper_bounds[valid_idx]
    lower_bounds = lower_bounds[valid_idx]
    assert upper_bounds.min() > 0, "Upper bounds must be positive"
    assert lower_bounds.min() > 0, "Lower bounds must be positive"
    
    N = len(data)
    
    # Initialize parameters
    weights = np.ones(n_components) / n_components
    mean = np.mean(np.log(data))
    means = np.random.normal(mean.item(), 2, n_components)
    sigma = np.std(np.log(data))
    sigmas = np.random.normal(sigma.item(), 1, n_components)
    
    log_data = np.log(data + 1e-10)  # Small epsilon for stability
    
    for _ in range(max_iter):
        old_means = np.copy(means)
        old_sigmas = np.copy(sigmas)
        
        # E-step: compute responsibilities
        resp = np.zeros((N, n_components))
        for k in range(n_components):
            a, b = np.log(lower_bounds), np.log(upper_bounds)
            prob = truncnorm.pdf(log_data, a, b, loc=means[k], scale=sigmas[k])
            resp[:, k] = weights[k] * prob
            if np.isnan(prob).any() or np.isinf(prob).any() or prob.sum() < 1e-10:
                resp[:, k] = weights[k] * 1e-10
            
        resp_sum = resp.sum(axis=1, keepdims=True)
        resp_sum[resp_sum < 1e-10] = 1e-10
        resp /= resp_sum
        
        if np.isnan(resp).any() or np.isinf(resp).any():
            print("NaN or Inf in responsibilities")
            import pdb; pdb.set_trace()
        
        # M-step: update parameters
        Nk = resp.sum(axis=0)
        Nk[Nk < 1e-10] = 1e-10
        weights = Nk / N
        
        for k in range(n_components):
            weighted_sum = np.sum(resp[:, k] * log_data)
            means[k] = weighted_sum / Nk[k]
            
            var_k = np.sum(resp[:, k] * (log_data - means[k])**2) / Nk[k]
            if np.isnan(var_k) or var_k < 1e-6:
                var_k = 1e-6
                
            sigmas[k] = np.sqrt(var_k)
            
        if np.all(np.abs(means - old_means) < 1e-6) and np.all(np.abs(sigmas - old_sigmas) < 1e-6):
            break
    
    for k in range(n_components):
        if np.isnan(means[k]) or np.isinf(means[k]):
            means[k] = 0.0
        if np.isnan(sigmas[k]) or np.isinf(sigmas[k]) or sigmas[k] < 1e-6:
            sigmas[k] = 1.0
    
    return weights, means, sigmas

def sample_truncated_lognormal_mixture(weights, means, sigmas, upper_bounds, n_samples=1):
    """Sample from truncated Log-normal mixture model"""
    samples = []
    for i in range(n_samples):
        k = np.random.choice(len(weights), p=weights)
        mean, sigma = means[k], sigmas[k]
        # a, b = (np.log(1e-10) - mean) / sigma, (np.log(upper_bounds) - mean) / sigma
        a, b = np.log(1e-10), np.log(upper_bounds[i])
        sample = np.exp(truncnorm.rvs(a, b, loc=mean, scale=sigma))
        samples.append(sample)
    
    return np.array(samples)

def sample_truncated_lognormal_mixture_v2(pi, mu, sigma, upper_bounds, lower_bounds):
    """
    Sample durations from a mixture of truncated lognormal distributions.

    Parameters:
    -----------
    pi : array-like
        Weights of the mixture components, shape (K,), where K is the number of components.
        Must sum to 1 and each element must be positive.
    mu : array-like
        Means of the underlying normal distributions for each component, shape (K,).
    sigma : array-like
        Standard deviations of the underlying normal distributions for each component, shape (K,).
        Must be positive.
    T : array-like
        Truncation points (e.g., gaps between consecutive visits), shape (N,), where N is the number of samples.
        Must be positive.
    lower_bounds : array-like
        Lower truncation points, shape (N,), where N is the number of samples.
        Must be positive.

    Returns:
    --------
    X : ndarray
        Sampled durations, shape (N * n_samples_per_truncation,), where each element is between the corresponding lower and upper bounds.

    Notes:
    ------
    - The function assumes a mixture of lognormal distributions, where each component is truncated from below at lower_bounds and above at T.
    - For each sample, a component is selected based on adjusted weights that account for the truncation,
      and then a duration is sampled from the selected component's truncated lognormal distribution.
    """
    # Convert inputs to numpy arrays
    pi = np.asarray(pi)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    upper_bounds = np.asarray(upper_bounds)
    lower_bounds = np.asarray(lower_bounds)
    
    # Get dimensions
    K = len(pi)  # Number of components
    N = len(upper_bounds)   # Number of samples
    
    # Compute log of truncation points
    log_ub = np.log(upper_bounds)
    log_lb = np.log(lower_bounds)
    
    # Compute CDF of each component up to each truncation point
    # Shape (K, N), where F[k, j] = P(X_k <= T[j]) for component k and truncation T[j]
    diff_upper = (log_ub[None, :] - mu[:, None]) / sigma[:, None]
    diff_lower = (log_lb[None, :] - mu[:, None]) / sigma[:, None]
    F_upper = norm.cdf(diff_upper)
    F_lower = norm.cdf(diff_lower)
    
    # Compute adjusted weights for component selection
    # pi_adj[k, j] is proportional to pi[k] * (P(X_k <= T[j]) - P(X_k <= lower_bounds[j]))
    pi_adj = pi[:, None] * (F_upper - F_lower)
    sum_pi_F = np.sum(pi_adj, axis=0)  # Normalization factor for each T[j]
    pi_adj /= sum_pi_F[None, :]        # Normalized probabilities, shape (K, N)
    
    # Sample component indices for each T[j]
    if np.any(np.isnan(pi_adj)) or np.any(np.isinf(pi_adj)):
        print("NaN or Inf in adjusted weights")
        import pdb; pdb.set_trace()
    k_samples = np.array([np.random.choice(K, p=pi_adj[:, j]) for j in range(N)])
    
    # Select parameters for the chosen components
    mu_selected = mu[k_samples]      # Shape (N,)
    sigma_selected = sigma[k_samples]  # Shape (N,)
    
    # Compute the CDF of the normal distribution up to log(T[j]) for each selected component
    p_upper = norm.cdf((log_ub - mu_selected) / sigma_selected)  # Shape (N,)
    p_lower = norm.cdf((log_lb - mu_selected) / sigma_selected)  # Shape (N,)
    
    # Sample from the truncated normal distributions
    W = np.random.rand(N)  # New uniform variables, shape (N,)
    Y = mu_selected + sigma_selected * norm.ppf(p_lower + W * (p_upper - p_lower))
    # Y[j] is a sample from N(mu[k], sigma[k]^2) truncated to (log(lower_bounds[j]), log(T[j]))
    
    # Transform back to lognormal scale
    X = np.exp(Y)  # Shape (N,)
    
    return X

def plot_truncated_lognormal_mixture(ax, data, weights, means, sigmas, upper_bounds, lower_bounds, color='red', show_samples=False, version=1):
    """Plot fitted truncated Log-normal mixture components and their sum"""
    x = np.linspace(min(data), max(data), 1000)
    total_pdf = np.zeros_like(x)
    
    for i in range(len(weights)):
        # a, b = (np.log(1e-10) - means[i]) / sigmas[i], (np.log(upper_bounds) - means[i]) / sigmas[i]
        # a, b = np.log(1e-10), np.log(upper_bounds)
        # component_pdf = weights[i] * truncnorm.pdf(np.log(x), a, b, loc=means[i], scale=sigmas[i])
        component_pdf = weights[i] * lognorm.pdf(x, s=sigmas[i], scale=np.exp(means[i]))
        total_pdf += component_pdf
        # Plot individual components
        ax.plot(x, component_pdf, '-', linewidth=0.8, color=color, alpha=0.3)
    
    # Plot total mixture
    ax.plot(x, total_pdf, color=color, label='Truncated Log-normal Mixture', linewidth=2)
    
    if show_samples:
        # Sample from the mixture and plot histogram
        if version == 1:
            samples = sample_truncated_lognormal_mixture(weights, means, sigmas, upper_bounds, n_samples=len(data))
        elif version == 2:
            samples = sample_truncated_lognormal_mixture_v2(weights, means, sigmas, upper_bounds, lower_bounds)
            
        ax.hist(samples, bins=50, density=True, color=color, alpha=0.3, label='Samples')
    
    # Calculate log likelihood and number of non-neglected components
    log_likelihood = np.sum(np.log(np.sum([weights[i] * truncnorm.pdf(np.log(data), (np.log(1e-10) - means[i]) / sigmas[i], (np.log(upper_bounds) - means[i]) / sigmas[i], loc=means[i], scale=sigmas[i]) for i in range(len(weights))], axis=0)))
    non_neglected_components = np.sum(weights > 1e-10)
    ax.set_xlabel(f'Log Likelihood: {log_likelihood:.2f}, Non-neglected Components: {non_neglected_components}')
    
    return total_pdf
    
def fit_dpm_bg_lognorm_mixture(data, max_components=10, alpha=0.1, max_iter=100):
    # fit Dirichlet Process Mixture of Log-normal distributions using BayesianGaussianMixture
    data = np.array(data)
    valid_idx = data > 0
    
    if np.sum(valid_idx) < len(data):
        print(f"Warning: Removed {len(data) - np.sum(valid_idx)} non-positive values for DPM log-normal fitting")
    
    if np.sum(valid_idx) < 5:  # If too few valid data points
        # Return default parameters
        return np.array([1.0]), np.array([0.0]), np.array([1.0])
    
    data = data[valid_idx]
    N = len(data)
    log_data = np.log(data)
    
    # fit using BayesianGaussianMixture
    dpgmm = BayesianGaussianMixture(
        n_components=max_components,  # generous upper bound
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=alpha,  # encourages sparsity in component usage
        max_iter=max_iter,
        covariance_type='diag',
        random_state=42
    )
    
    dpgmm.fit(log_data.reshape(-1, 1))
    dp_weights = dpgmm.weights_
    dpgmm_n_components = np.sum(dp_weights > 1e-2)
    dp_means = dpgmm.means_.flatten()  # means in log-space
    dp_variances = np.array([dpgmm.covariances_[k, 0] for k in range(dpgmm.n_components)])
    dp_sigmas = np.sqrt(dp_variances)
    return dp_weights, dp_means, dp_sigmas

def fit_dpm_truncated_lognormal_mixture(data, upper_bounds, lower_bounds, max_components=10, alpha=0.1, max_iter=100):
    """Fit Dirichlet Process Mixture of truncated Log-normal distributions using BayesianGaussianMixture"""
    # Filter out non-positive values (log is undefined for <= 0)
    data = np.array(data)
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)
    valid_idx = data > 0
    
    if np.sum(valid_idx) < len(data):
        logging.info(f"Warning: Removed {len(data) - np.sum(valid_idx)} non-positive values for DPM log-normal fitting")
    
    if np.sum(valid_idx) < 10:  # If too few valid data points
        # Return default parameters
        return np.array([1.0]), np.array([0.0]), np.array([1.0])
    
    data = data[valid_idx]
    lower_bounds = lower_bounds[valid_idx]
    upper_bounds = upper_bounds[valid_idx]
    N = len(data)
    log_data = np.log(data)
    
    # Initialize with K-means
    K = min(max_components, N // 10 + 1)  # Ensure we have enough data points per component
    kmeans = KMeans(n_clusters=K, random_state=42).fit(log_data.reshape(-1, 1))
    
    # Initialize mixture parameters
    cluster_assignments = kmeans.labels_
    unique_clusters = np.unique(cluster_assignments)
    K_active = len(unique_clusters)
    
    # Initialize component parameters (mean, precision)
    components = []
    weights = np.zeros(K_active)
    
    for k, cluster_id in enumerate(unique_clusters):
        cluster_data = log_data[cluster_assignments == cluster_id]
        mean = np.mean(cluster_data)
        var = np.var(cluster_data)
        if var < 1e-6:
            var = 1e-6
        precision = 1.0 / var
        components.append([mean, precision])
        weights[k] = len(cluster_data) / N
    
    # Convert to numpy array
    components = np.array(components)
    
    # EM for DP mixture
    for _ in range(max_iter):
        # E-step: compute responsibilities
        log_resp = np.zeros((N, K_active))
        
        for k in range(K_active):
            mean, precision = components[k]
            variance = 1.0 / precision
            a, b = (np.log(lower_bounds) - mean) / np.sqrt(variance), (np.log(upper_bounds) - mean) / np.sqrt(variance)
            truncnorm_cdf_diff = truncnorm.cdf(b, a, b) - truncnorm.cdf(a, a, b)
            log_resp[:, k] = np.log(weights[k] + 1e-10) - 0.5 * np.log(2 * np.pi * variance) - \
                            0.5 * precision * (log_data - mean)**2 - np.log(truncnorm_cdf_diff + 1e-10)
        
        # Normalize responsibilities
        log_resp_max = np.max(log_resp, axis=1, keepdims=True)
        resp = np.exp(log_resp - log_resp_max)
        resp_sum = resp.sum(axis=1, keepdims=True)
        resp_sum[resp_sum < 1e-10] = 1e-10
        resp = resp / resp_sum
        
        # M-step
        Nk = resp.sum(axis=0)
        active_components = Nk > (alpha / K_active)
        
        if np.sum(active_components) == 0:
            # If no active components, keep the strongest one
            active_components[np.argmax(Nk)] = True
        
        # Update weights with Dirichlet prior
        weights = (Nk + alpha/K_active) / (N + alpha)
        
        # Update components for active clusters
        new_components = []
        new_weights = []
        
        for k in range(K_active):
            if active_components[k]:
                # Update mean and precision for active component
                resp_k = resp[:, k]
                weighted_sum = np.sum(resp_k * log_data)
                weighted_mean = weighted_sum / (Nk[k] + 1e-10)
                
                weighted_var = np.sum(resp_k * (log_data - weighted_mean)**2) / (Nk[k] + 1e-10)
                if weighted_var < 1e-6:
                    weighted_var = 1e-6
                
                weighted_precision = 1.0 / weighted_var
                
                new_components.append([weighted_mean, weighted_precision])
                new_weights.append(weights[k])
        
        # Update active components
        components = np.array(new_components)
        weights = np.array(new_weights)
        weights = weights / np.sum(weights)  # Renormalize weights
        K_active = len(components)
        
        # Check for convergence if K_active is stable
        if K_active <= 1:
            break
    
    means = components[:, 0]
    precision = np.sqrt(1.0 / components[:, 1])
    
    return weights, means, precision

def sample_dpm_truncated_lognormal_mixture(weights, means, sigmas, upper_bounds, lower_bounds, n_samples=1):
    """Sample from Dirichlet Process Mixture of truncated Log-normal distributions"""
    samples = []
    for i in range(n_samples):
        # Sample a component
        k = np.random.choice(len(weights), p=weights)
        mean, sigma = means[k], sigmas[k]
        a, b = (np.log(lower_bounds[i]) - mean) / sigma, (np.log(upper_bounds[i]) - mean) / sigma
        sample = np.exp(truncnorm.rvs(a, b, loc=mean, scale=sigma))
        samples.append(sample)
    
    return np.array(samples)

def sample_dpm_truncated_lognormal_mixture_v2(weights, means, sigmas, upper_bounds, lower_bounds):
    """Sample from Dirichlet Process Mixture of truncated Log-normal distributions with adjusted weights
    similar to sample_truncated_lognormal_mixture_v2"""
    # Convert inputs to numpy arrays
    pi = np.asarray(weights)
    mu = np.asarray(means)
    sigma = np.asarray(sigmas)
    upper_bounds = np.asarray(upper_bounds)
    lower_bounds = np.asarray(lower_bounds)
    
    # Get dimensions
    K = len(pi)  # Number of components
    N = len(upper_bounds)   # Number of samples
    
    # Compute log of truncation points
    log_ub = np.log(upper_bounds)
    log_lb = np.log(lower_bounds)
    
    # Compute CDF of each component up to each truncation point
    # Shape (K, N), where F[k, j] = P(X_k <= T[j]) for component k and truncation T[j]
    diff_upper = (log_ub[None, :] - mu[:, None]) / sigma[:, None]
    diff_lower = (log_lb[None, :] - mu[:, None]) / sigma[:, None]
    F_upper = norm.cdf(diff_upper)
    F_lower = norm.cdf(diff_lower)
    
    # Compute adjusted weights for component selection
    # pi_adj[k, j] is proportional to pi[k] * (P(X_k <= T[j]) - P(X_k <= lower_bounds[j]))
    pi_adj = pi[:, None] * (F_upper - F_lower)
    sum_pi_F = np.sum(pi_adj, axis=0)  # Normalization factor for each T[j]
    pi_adj /= sum_pi_F[None, :]        # Normalized probabilities, shape (K, N)
    
    # Sample component indices for each T[j]
    if np.any(np.isnan(pi_adj)) or np.any(np.isinf(pi_adj)):
        print("NaN or Inf in adjusted weights")
        import pdb; pdb.set_trace()
        
    k_samples = np.array([np.random.choice(K, p=pi_adj[:, j]) for j in range(N)])
    
    # Select parameters for the chosen components
    mu_selected = mu[k_samples]      # Shape (N,)
    sigma_selected = sigma[k_samples]  # Shape (N,)
    
    # Compute the CDF of the normal distribution up to log(T[j]) for each selected component
    p_upper = norm.cdf((log_ub - mu_selected) / sigma_selected)  # Shape (N,)
    p_lower = norm.cdf((log_lb - mu_selected) / sigma_selected)  # Shape (N,)
    
    # Sample from the truncated normal distributions
    W = np.random.rand(N)  # New uniform variables, shape (N,)
    Y = mu_selected + sigma_selected * norm.ppf(p_lower + W * (p_upper - p_lower))
    # Y[j] is a sample from N(mu[k], sigma[k]^2) truncated to (log(lower_bounds[j]), log(T[j]))
    
    # Transform back to lognormal scale
    X = np.exp(Y)  # Shape (N,)
    
    return X
    

def plot_dpm_truncated_lognormal_mixture(ax, data, weights, means, sigmas, upper_bounds, lower_bounds, color='blue', show_samples=False, version=1):
    """Plot fitted Dirichlet Process Mixture of truncated Log-normal distributions"""
    x = np.linspace(min(data), max(data), 1000)
    total_pdf = np.zeros_like(x)
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)
    
    for i in range(len(weights)):
        # mean, sigma = means[i], sigmas[i]
        # a, b = (np.log(lower_bounds) - mean) / sigma, (np.log(upper_bounds) - mean) / sigma
        component_pdf = weights[i] * lognorm.pdf(x, s=sigmas[i], scale=np.exp(means[i]))
        total_pdf += component_pdf
        # Plot individual components
        ax.plot(x, component_pdf, '-', linewidth=0.8, color=color, alpha=0.3)
    
    # Plot total mixture
    ax.plot(x, total_pdf, color=color, label='DPM Truncated Log-normal', linewidth=0.8)
    samples = None # Placeholder for samples
    if show_samples:
        # Sample from the mixture and plot histogram
        if version == 1:
            samples = sample_dpm_truncated_lognormal_mixture(weights, means, sigmas, upper_bounds, lower_bounds)
        elif version == 2:
            samples = sample_dpm_truncated_lognormal_mixture_v2(weights, means, sigmas, upper_bounds, lower_bounds)
            
        ax.hist(samples, bins=50, density=True, color=color, alpha=0.3, label='Samples')
    
    # Calculate log likelihood and number of non-neglected components
    ln_lb = (np.log(lower_bounds[:len(data)]) - means[i]) / sigmas[i]
    ln_ub = (np.log(upper_bounds[:len(data)]) - means[i]) / sigmas[i]
    log_likelihood = np.sum(np.log(
                     np.sum([weights[i] * truncnorm.pdf(
                                            np.log(data), ln_lb, ln_ub, 
                                            loc=means[i], scale=sigmas[i]
                                        ) 
                        for i in range(len(weights))], 
                        axis=0
                    )))
    
    non_neglected_components = np.sum(weights > 1e-3)
    ax.set_xlabel(f'Log Likelihood: {log_likelihood:.2f}, Non-neglected Components: {non_neglected_components}')
    
    return log_likelihood, samples


def plot_pca_node_embeddings(node_embeddings, labels, save_path):
    """
    Plot PCA of node embeddings to visualize k-means clustering.

    Parameters:
    -----------
    node_embeddings : ndarray
        Node embeddings, shape (N, D), where N is the number of nodes and D is the embedding dimension.
    labels : array-like
        Cluster labels for each node, shape (N,).
    save_path : str
        Path to save the PCA plot.
    """
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(node_embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA of Node Embeddings')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"PCA plot saved in {save_path}")

from sklearn.manifold import TSNE
import umap

def plot_tsne_node_embeddings(node_embeddings, labels, save_path):
    """
    Plot t-SNE of node embeddings to visualize k-means clustering.

    Parameters:
    -----------
    node_embeddings : ndarray
        Node embeddings, shape (N, D), where N is the number of nodes and D is the embedding dimension.
    labels : array-like
        Cluster labels for each node, shape (N,).
    save_path : str
        Path to save the t-SNE plot.
    """
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(node_embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE of Node Embeddings')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"t-SNE plot saved in {save_path}")

def plot_umap_node_embeddings(node_embeddings, labels, save_path):
    """
    Plot UMAP of node embeddings to visualize k-means clustering.

    Parameters:
    -----------
    node_embeddings : ndarray
        Node embeddings, shape (N, D), where N is the number of nodes and D is the embedding dimension.
    labels : array-like
        Cluster labels for each node, shape (N,).
    save_path : str
        Path to save the UMAP plot.
    """
    reducer = umap.UMAP(random_state=42, n_components=2)
    umap_result = reducer.fit_transform(node_embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(umap_result[:, 0], umap_result[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.title('UMAP of Node Embeddings')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"UMAP plot saved in {save_path}")

from mpl_toolkits.mplot3d import Axes3D
from joblib import Parallel, delayed

def plot_pca_node_embeddings_3d(node_embeddings, labels, save_path):
    """
    Plot 3D PCA of node embeddings to visualize k-means clustering.

    Parameters:
    -----------
    node_embeddings : ndarray
        Node embeddings, shape (N, D), where N is the number of nodes and D is the embedding dimension.
    labels : array-like
        Cluster labels for each node, shape (N,).
    save_path : str
        Path to save the 3D PCA plot.
    """
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(node_embeddings)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=labels, cmap='viridis', alpha=0.5)
    fig.colorbar(scatter, ax=ax, label='Cluster')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    ax.set_title('3D PCA of Node Embeddings')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"3D PCA plot saved in {save_path}")

def plot_tsne_node_embeddings_3d(node_embeddings, labels, save_path):
    """
    Plot 3D t-SNE of node embeddings to visualize k-means clustering.

    Parameters:
    -----------
    node_embeddings : ndarray
        Node embeddings, shape (N, D), where N is the number of nodes and D is the embedding dimension.
    labels : array-like
        Cluster labels for each node, shape (N,).
    save_path : str
        Path to save the 3D t-SNE plot.
    """
    tsne = TSNE(n_components=3, random_state=42)
    tsne_result = tsne.fit_transform(node_embeddings)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1], tsne_result[:, 2], c=labels, cmap='viridis', alpha=0.5)
    fig.colorbar(scatter, ax=ax, label='Cluster')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    ax.set_title('3D t-SNE of Node Embeddings')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"3D t-SNE plot saved in {save_path}")

def plot_umap_node_embeddings_3d(node_embeddings, labels, save_path):
    """
    Plot 3D UMAP of node embeddings to visualize k-means clustering.

    Parameters:
    -----------
    node_embeddings : ndarray
        Node embeddings, shape (N, D), where N is the number of nodes and D is the embedding dimension.
    labels : array-like
        Cluster labels for each node, shape (N,).
    save_path : str
        Path to save the 3D UMAP plot.
    """
    reducer = umap.UMAP(random_state=42, n_components=3)
    umap_result = reducer.fit_transform(node_embeddings)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(umap_result[:, 0], umap_result[:, 1], umap_result[:, 2], c=labels, cmap='viridis', alpha=0.5)
    fig.colorbar(scatter, ax=ax, label='Cluster')
    ax.set_xlabel('UMAP Component 1')
    ax.set_ylabel('UMAP Component 2')
    ax.set_zlabel('UMAP Component 3')
    ax.set_title('3D UMAP of Node Embeddings')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"3D UMAP plot saved in {save_path}")



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


def viz(hcw_embs, hcw_cluster_durations, hcw_cluster_upperbounds, hcws_by_label_cluster, num_hcw_clusters, min_duration, pp_save_dir):
    """Visualize HCW cluster durations using different distributions"""
    # fitting mixture of gamma distribution to the durations
    # Select 3 HCWs per label for visualization
    labels_to_viz = sorted(hcw_cluster_durations.keys())
    viz_data = {}
    for label in labels_to_viz:
        clusters = list(hcw_cluster_durations[label].keys())[:3]  # Take first 3 clusters per label
        viz_data[label] = clusters
    
    viz_node_embs = False
    if viz_node_embs:
        # Plot PCA of node embeddings
        pca_save_path = pp_save_dir + "/pca_node_embeddings.png"
        plot_pca_node_embeddings(hcw_embs, hcw_clusters, pca_save_path)
        
        # Plot t-SNE of node embeddings
        tsne_save_path = pp_save_dir + "/tsne_node_embeddings.png"
        plot_tsne_node_embeddings(hcw_embs, hcw_clusters, tsne_save_path)
        
        # Plot UMAP of node embeddings
        umap_save_path = pp_save_dir + "/umap_node_embeddings.png"
        plot_umap_node_embeddings(hcw_embs, hcw_clusters, umap_save_path)

        # Plot 3D PCA of node embeddings
        pca_save_path_3d = pp_save_dir + "/pca_node_embeddings_3d.png"
        plot_pca_node_embeddings_3d(hcw_embs, hcw_clusters, pca_save_path_3d)
        
        # Plot 3D t-SNE of node embeddings
        tsne_save_path_3d = pp_save_dir + "/tsne_node_embeddings_3d.png"
        plot_tsne_node_embeddings_3d(hcw_embs, hcw_clusters, tsne_save_path_3d)
        
        # Plot 3D UMAP of node embeddings
        umap_save_path_3d = pp_save_dir + "/umap_node_embeddings_3d.png"
        plot_umap_node_embeddings_3d(hcw_embs, hcw_clusters, umap_save_path_3d)


    pbar =  tqdm(['dpm_truncated_lognormal'], desc="Fitting distributions")
    for fit_dist in pbar:
        pbar.set_description(f"Fitting {fit_dist} distributions")
        
        total_plots = sum(len(hcws) for hcws in viz_data.values())
        fig, axs = plt.subplots(total_plots, 1, figsize=(15, 5*total_plots))
        
        plot_idx = 0
        for label in tqdm(labels_to_viz):
            for cluster in viz_data[label]:
                durations = hcw_cluster_durations[label][cluster]
                upper_bounds = hcw_cluster_upperbounds[label][cluster]
                lower_bounds = np.ones_like(upper_bounds) * min_duration
                if not durations or len(durations) < 5:  # Skip if insufficient data
                    continue
                    
                ax = axs[plot_idx]
                sns.histplot(durations, bins=100, stat='density', alpha=0.5, ax=ax, label='Histogram')
                sns.kdeplot(durations, ax=ax, label='KDE')
                if fit_dist == 'dpm_truncated_lognormal':
                    weights_tl, means_tl, sigmas_tl = fit_dpm_truncated_lognormal_mixture(durations, upper_bounds, lower_bounds, max_components=10)
                    plot_dpm_truncated_lognormal_mixture(ax, durations, weights_tl, means_tl, sigmas_tl, upper_bounds, lower_bounds, show_samples=True)
                    
                if fit_dist == 'truncated_lognormal_v2':
                    lower_bounds = np.zeros_like(upper_bounds)
                    weights_tl, means_tl, sigmas_tl = fit_truncated_lognormal_mixture(durations, upper_bounds, lower_bounds, n_components=10)
                    plot_truncated_lognormal_mixture(ax, durations, weights_tl, means_tl, sigmas_tl, upper_bounds, lower_bounds, show_samples=True, version=2)
                
                ax.set_title(f"Label {label} - HCW cluster {cluster} ({hcws_by_label_cluster[label][cluster]})")
                ax.legend()
                plot_idx += 1
                
        plt.tight_layout()
        figsavepath = pp_save_dir + f"/durations_viz_{fit_dist}_{num_hcw_clusters}clusters.png"
        plt.savefig(figsavepath)
        logging.info("Duration visualization saved in " + str(Path(figsavepath).absolute()))
        plt.close()


from metrics.metric_utils import (
    get_unique_string_from_edge_tuple, 
    get_string_from_edge_tuple, 
    convert_graph_from_defauldict_to_dict
)

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
            # Resolve conflict by keeping the visit with maximum count
            if count > resolved_visits[-1][1]:
                resolved_visits[-1] = visit
                last_end_time = start_time
    
    return resolved_visits


def major_voting_snapshots_src_dst_pairs(adj_matrix_temporal_sampled, min_day, max_day, time_window, 
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

def _check(hcw_std_graph, daily_hcw_room_pairs, undirected=True):
    found_daily_pairs_check = defaultdict(set)
    for hcw in hcw_std_graph:
        for i in range(len(hcw_std_graph[hcw])):
            room, cnt, start_time = hcw_std_graph[hcw][i][:3]
            day_idx = start_time // one_day_window
            found_daily_pairs_check[day_idx].add((hcw, room))
    
    target_daily_pairs = daily_hcw_room_pairs
            
    for start_time in range(min_t, max_t, one_day_window):
        day_idx = start_time // one_day_window
        if target_daily_pairs[day_idx] < found_daily_pairs_check[day_idx]:
            logging.info("Mismatch in daily pairs for day_idx: ", day_idx)
            logging.info(f"Target daily pairs ({len(target_daily_pairs[day_idx])}): ", target_daily_pairs[day_idx])
            logging.info(f"Found daily pairs ({len(found_daily_pairs_check[day_idx])}): ", found_daily_pairs_check[day_idx])
            return False, found_daily_pairs_check, target_daily_pairs
        
    return True, found_daily_pairs_check, target_daily_pairs


def fit_distribution_for_cluster(label, cluster, durations, 
                                 upper_bounds, lower_bounds, 
                                 max_components=10, alpha=0.1, 
                                 max_iter=1000):
    if not durations or len(durations) < 5: return None
    # weights_tl, means_tl, sigmas_tl = fit_truncated_lognormal_mixture(durations, upper_bounds, lower_bounds, n_components=10)
    weights_tl, means_tl, sigmas_tl = fit_dpm_bg_lognorm_mixture(durations, 
                                            max_components=max_components, 
                                            alpha=alpha, max_iter=max_iter)
    
    return (label, cluster, weights_tl, means_tl, sigmas_tl)


def fuse_events(event_list, fuse_gap):
    '''
    Each event is (cnt, time) in sorted order of time 
    '''
    fused_events = []
    current_group = []
    current_time = None
    
    for event in event_list:
        time = event[1]
        
        if current_time is None or (time - current_time <= fuse_gap and time - current_group[0][1] <= fuse_gap):
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
                       all_fitted_dists, min_duration, max_duration, hcw_to_cluster,
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
    sampled_walks = sequences_from_temporal_walks(generated_events, generated_times, end_node_id, max_t, inv_vocab, l_w)
    adj_list_temporal_sampled = get_adj_graph_from_random_walks(sampled_walks, min_t, max_t, True)
    
    # from now on, hcw/room ids are actual ids, not the vocab id
    
    # extract graph with major-voting conflict resolution 
    logging.info(f"\nExtracting graph for label {label} with major-voting conflict resolution")
    
    # Step 1: extract daily-based set of hcw-room pairs
    _, sampled_graphs = major_voting_snapshots_src_dst_pairs(
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
    
    # assert no duplicate (room, time) pairs inside hcw_std_graph[hcw]
    for hcw in hcw_std_graph:
        if not (len(hcw_std_graph[hcw]) == len(set(hcw_std_graph[hcw]))): 
            import pdb; pdb.set_trace()
        
    # sort visits based on start time
    for hcw in hcw_std_graph:
        hcw_std_graph[hcw] = sorted(hcw_std_graph[hcw], key=lambda x: x[2])
        
    # resolve conflicts
    hcw_std_graph_resolved = {}
    for hcw in hcw_std_graph:
        resolved_visits = resolve_conflicts(hcw_std_graph[hcw], fuse_gap)
        hcw_std_graph_resolved[hcw] = resolved_visits
    hcw_std_graph = hcw_std_graph_resolved
    
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
        cluster = hcw_to_cluster[vocab[hcw]]
        weights_tl, means_tl, sigmas_tl = all_fitted_dists[cluster]
        ubs = hcw_vists_gaps[hcw]
        ubs = np.array(ubs) - slack_gap  # subtract slack_gap seconds for realistic transition
        ubs = np.minimum(ubs, max_duration)
        assert np.all(ubs > min_duration)
        
        lbs = np.ones_like(ubs) * min_duration
        durations = sample_truncated_lognormal_mixture_v2(weights_tl, means_tl, sigmas_tl, ubs, lbs)
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
    
    pp_save_dir = str(Path(config_dir) / savedir)
    Path(pp_save_dir).mkdir(parents=True, exist_ok=True)

    strtime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    
    train_logfilepath = glob.glob(str(Path(config_dir)/ 'train*~log.txt'))[-1]
    if not Path(train_logfilepath).exists():
        logging.info("Training log file not found")
        os._exit(os.EX_OK)

    eval_start_time = time.time()
    PROJECT_NAME = "ECMLPKDD25_TIGGER"
    run_name = str(Path(train_logfilepath.replace('train',f'postpro~{strtime}')).stem)[:-4]

    logfilepath = config_logging(args, pp_save_dir, run_name, PROJECT_NAME, use_wandb=False)
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
            batch_size=batch_size,
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

    # hcws_by_label = defaultdict(list)
    
    # for label in graph_label_to_id.values():
    #     hcw_rawids_by_lb = data[data['label'] == label]['start'].unique()
    #     # convert to id 
    #     hcws_by_label[label] = [vocab[hcw] for hcw in hcw_rawids_by_lb]
        
    fitdist_savepath = pp_save_dir + "/hcw_cluster_dist.pkl"
    
    if Path(fitdist_savepath).exists() and False:
        with open(fitdist_savepath, "rb") as f:
            dpgmm = pickle.load(f)
        logging.info("Fitted distributions loaded from " + str(fitdist_savepath))
    else:
        dpgmm = BayesianGaussianMixture(
            n_components=30,  # Upper bound on the number of components
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=1e-2,  # Encourages sparsity in component usage
            max_iter=1000,
            random_state=42
        )            
        hcw_embs = node_embs[hcws]
        dpgmm.fit(hcw_embs)
        # save the fitted distributions
        with open(fitdist_savepath, "wb") as f:
            pickle.dump(dpgmm, f)
        logging.info("Fitted distributions saved in " + str(fitdist_savepath))
    
    # predict cluster
    hcw_to_cluster = dict()
    hcw_embs = node_embs[hcws]
    hcw_clusters = dpgmm.predict(hcw_embs)
    num_hcw_clusters = np.sum(dpgmm.weights_ > 1e-2)
    logging.info("Number of HCW clusters: {}".format(num_hcw_clusters))
    for hcw, cluster in zip(hcws, hcw_clusters):
        hcw_to_cluster[hcw] = cluster
        
    # viz the durations
    hcw_cluster_durations = defaultdict(list)
    hcw_cluster_upperbounds = defaultdict(list)
    hcw_cluster_nvisits = defaultdict(int)
    cluster_to_hcws = defaultdict(set)
    for i, row in data.iterrows():
        lb, hcw = row['label'], vocab[row['start']]
        duration, gap = row['d'], row['next_start'] - row['days']
        # convert to int 
        duration, gap = int(duration), int(gap)
        
        hcw_cluster = hcw_to_cluster[hcw]
        hcw_cluster_durations[hcw_cluster].append(row['d'])
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
    
    # viz(hcw_embs, hcw_cluster_durations, hcw_cluster_upperbounds, hcws_by_label_cluster, num_hcw_clusters, min_duration, pp_save_dir)

    # fitting truncated_lognormal distributions to durations
    dur_dist_savepath = pp_save_dir + "/fitted_cluster_duration_dists.pkl"

    if Path(dur_dist_savepath).exists():
        with open(dur_dist_savepath, "rb") as f:
            all_fitted_dists = pickle.load(f)
        logging.info("Fitted distributions loaded from " + str(dur_dist_savepath))
    else:
        all_fitted_dists = dict()
        logging.info("Fitting lognormal distributions to durations")
        cluster_duration_info = []
        for cluster in hcw_cluster_durations:
            durations = hcw_cluster_durations[cluster]
            upper_bounds = hcw_cluster_upperbounds[cluster]
            lower_bounds = np.ones_like(upper_bounds) * min(min(durations), 30)
            cluster_duration_info.append((lb, cluster, durations, upper_bounds, lower_bounds))
                
        results = Parallel(n_jobs=10)(delayed(fit_distribution_for_cluster)(
            label, cluster, durations, upper_bounds, lower_bounds
        ) for label, cluster, durations, upper_bounds, lower_bounds 
            in tqdm(cluster_duration_info, desc="Fitting distributions"))
        
        for idx, result in enumerate(results):
            if not result: continue
            lb, cluster, weights_tl, means_tl, sigmas_tl = result
            all_fitted_dists[cluster] = (weights_tl, means_tl, sigmas_tl)
                
        # save the fitted distributions
        with open(dur_dist_savepath, "wb") as f:
            pickle.dump(all_fitted_dists, f)
        logging.info("Fitted distributions saved in " + str(dur_dist_savepath))
    #
    
    #
    min_t = 1
    max_t = int(data['days'].max())
    one_day_window = 60 * 60 * 24
    slack_gap = 30
    # fuse_gap = max(min(cluster_min_gaps.values()), 60 * 2) # 2 minutes
    fuse_gap = 60 * 2 # 2 minutes
    min_duration = min(cluster_min_duration.values())
    max_duration = max(cluster_max_duration.values())
    logging.info("min_t: {}, max_t: {}".format(min_t, max_t))
    logging.info("Fuse gap: {}, min_duration: {}, max_duration: {}".format(
                    fuse_gap, min_duration, max_duration))
    #
    if False:
        all_llhs = []
        for cluster in hcw_cluster_durations:
            # lb, cluster, weights_tl, means_tl, sigmas_tl = result
            weights_tl, means_tl, sigmas_tl = all_fitted_dists[cluster]
            # # visualize the histogram and kde of durations
            durations = hcw_cluster_durations[cluster]
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            ax.set_title(f"HCW cluster {cluster} ({len(cluster_to_hcws[cluster])} HCWs, {len(durations)} visits)")
            sns.histplot(durations, bins=50, stat='density', kde=True, alpha=0.5, ax=ax, label='Data histogram')
            srate = 2
            ubs = hcw_cluster_upperbounds[cluster]
            ubs = np.minimum(ubs, max_duration)
            assert np.all(ubs >= min_duration)
            
            lbs = np.ones_like(ubs) * min(min(durations), 30)
            
            llh_bg, samples_bg = plot_dpm_truncated_lognormal_mixture(ax, durations, weights_tl, means_tl, sigmas_tl, 
                                                                        list(ubs)*srate, list(lbs)*srate, color='green', 
                                                                        show_samples=True, version=2)
            all_llhs.append(llh_bg)
            # visualize the weights
            ax2 = ax.inset_axes([0.6, 0.4, 0.3, 0.3])
            ax2.bar(np.arange(len(weights_tl)), weights_tl, color='green', alpha=0.5)
            ax2.set_title('Mixture weights')
            ax2.grid(True)
            ax.set_xlabel('Duration (seconds)')
            ax.legend()
            plt.tight_layout()
            figsavepath = pp_save_dir + f"/viz/durations_viz_{cluster}.png"
            Path(figsavepath).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(figsavepath)
            plt.close()
        
        #
        # save all_llhs
        with open(pp_save_dir + "/all_llhs.json", "w") as f:
            json.dump(all_llhs, f)
        logging.info("Average log likelihood of uniform mixture: ", np.mean(all_llhs))


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
                          all_fitted_dists, min_duration, max_duration, hcw_to_cluster,
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