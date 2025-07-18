import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions using the alias method.

    The alias method allows for efficient sampling from a discrete probability distribution with many outcomes.
    This implementation is based on the description provided in the blog post by Keith Schwarz:
    "The Alias Method: Efficient Sampling with Many Discrete Outcomes" (https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/).

    "Notice that it has \mathcal{O}(K) time complexity for setup (computing the CDF) and \mathcal{O}(K) time complexity per sample. The per-sample complexity could probably be reduced to \mathcal{O}(\log K) with a better data structure for finding the threshold. It turns out, however, that we can do better and get \mathcal{O}(1) for the sampling, while still being \mathcal{O}(K).
    One such method is due to Kronmal and Peterson (1979) and is called the alias method. It is also described in the wonderful book by Devroye (1986). George Dahl, Hugo Larochelle and I used this method in our ICML paper on learning undirected n-gram models with large vocabularies."

    Parameters:
    probs (list or numpy.ndarray): A list or array of probabilities for the discrete outcomes.
                                   The probabilities should sum to 1.

    Returns:
    tuple: Two numpy arrays, J and q, which are used for sampling from the distribution.
           - J (numpy.ndarray): An array of indices used in the alias method.
           - q (numpy.ndarray): An array of probabilities used in the alias method.
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def sort_dict(diction):
    diction = [(key, value) for key, value in diction.items()]
    diction.sort(key=lambda val: val[1], reverse=True)
    return diction


# sort_dict(prods_new)
def print_incoming_outcoming_edges_of_edge(edge):

    print("Edge, ", edge)
    print("Incoming")
    for item in edge.incoming_edges:
        print(item)

    print("Outgoing")
    for item in edge.outgoing_edges:
        print(item)
    return


def sort_edges_timewise(edges, reverse):
    edges.sort(key=lambda val: val.time, reverse=reverse)
    return edges


def print_list_of_edges(edges, cut_off=100):
    print("###")
    for index, edge in enumerate(edges):
        if index < cut_off:
            print(index, edge)
    print("###")


def prepare_alias_table(edge, incoming=False, window_interactions=10):
    if not incoming:
        time_diffs = [item.time - edge.time for item in edge.outgoing_edges]
    else:
        time_diffs = [item.time - edge.time for item in edge.incoming_edges]
    mn = np.mean(time_diffs)
    std = np.std(time_diffs)
    if len(time_diffs) == 1 or std == 0:
        std = 1
        # print(time_diffs,[-(item - mn)/std for item in time_diffs] )
    time_diffs = [
        -(item - mn) / std for item in time_diffs
    ]  ### less time diff edge should be more prioritized
    time_diffs = np.exp(time_diffs)
    norm_const = sum(time_diffs)
    nbr_sample_probs = [float(prob) / norm_const for prob in time_diffs]
    J, q = alias_setup(nbr_sample_probs)
    return nbr_sample_probs, J, q


def print_incoming_outcoming_edges_of_edge(edge):

    print("Edge, ", edge)
    print("Incoming")
    for item in edge.incoming_edges:
        print(item)

    print("Outgoing")
    for item in edge.outgoing_edges:
        print(item)
    return


def sort_edges_timewise(edges, reverse):
    edges.sort(key=lambda val: val.time, reverse=reverse)
    return edges


def binary_search_find_time_greater_equal(arr, target, strictly=False):
    start = 0
    end = len(arr) - 1
    ans = -1
    while start <= end:
        mid = (start + end) // 2

        # Move to right side if target is
        # greater.
        if strictly:
            if arr[mid].time <= target:
                start = mid + 1
            else:
                ans = mid
                end = mid - 1
        else:
            if arr[mid].time < target:
                start = mid + 1
            else:
                ans = mid
                end = mid - 1
        # Move left side.
    if not strictly:  ### find the first occurrance of this target
        less_found = False
        while ans != -1 and ans > 0 and not less_found:
            if arr[ans - 1].time == target:
                ans = ans - 1
            else:
                less_found = True
    return ans


def binary_search_find_time_lesser_equal(arr, target, strictly=False):
    if len(arr) == 0: return -1
    
    if arr[-1].time < target:
        return len(arr) - 1
    index = binary_search_find_time_greater_equal(arr, target, strictly=False)
    # print(index)
    if index == -1:
        return index
    if strictly:
        return index - 1
    else:

        if arr[index].time == target:
            return index
        else:
            return index - 1


# binary_search_find_time_lesser_equal(start_node_edges,44623,True)
# binary_search_find_time_greater_equal(start_node_edges,-1,True)


class Edge:
    def __init__(self, start, end, **kwargs):
        self.start = start
        self.end = end
        self.__dict__.update(kwargs)

    def __str__(self):
        s = "start: " + str(self.start) + " end: " + str(self.end) + " "
        if "time" in self.__dict__:
            s += "time: " + str(self.__dict__["time"])
        return s


class Node:
    def __init__(self, id, **kwargs):
        self.id = id
        self.__dict__.update(kwargs)


def prepare_alias_table_for_edge(edge, incoming=False, window_interactions=None):
    if not incoming:
        if window_interactions is None:
            window_interactions = len(edge.outgoing_edges)
        time_diffs = [
            item.time - edge.time for item in edge.outgoing_edges[:window_interactions]
        ]
    else:
        if window_interactions is None:
            window_interactions = len(edge.incoming_edges)
        time_diffs = [
            item.time - edge.time for item in edge.incoming_edges[:window_interactions]
        ]
    mn = np.mean(time_diffs)
    std = np.std(time_diffs)
    if len(time_diffs) == 1 or std == 0:
        std = 1
        # print(time_diffs,[-(item - mn)/std for item in time_diffs] )
    time_diffs = [
        -(item - mn) / std for item in time_diffs
    ]  ### less time diff edge should be more prioritized
    time_diffs = np.exp(time_diffs)
    norm_const = sum(time_diffs)
    nbr_sample_probs = [float(prob) / norm_const for prob in time_diffs]
    J, q = alias_setup(nbr_sample_probs)
    return nbr_sample_probs, J, q


def run_random_walk_without_temporal_constraints(edge, max_length=20, delta=0):
    rw = []
    # max_length = random.randint(20,50)
    if len(edge.incoming_edges) > 0:  ######### Add event for
        random_walk_start_time = edge.incoming_edges[
            alias_draw(edge.inJ, edge.inq)
        ].time
    else:
        random_walk_start_time = edge.time - delta
    random_walk = [edge]
    ct = 1
    done = False
    while ct < max_length and not done:
        if len(edge.out_nbr_sample_probs) == 0:
            done = True
            random_walk.append(
                Edge(start=edge.end, end="end_node", time=edge.time, label=edge.label)
            )
        else:
            tedge = edge.outgoing_edges[alias_draw(edge.outJ, edge.outq)]
            edge = tedge
            random_walk.append(edge)
            ct += 1
    return [random_walk_start_time] + [
        (edge.start, edge.end, edge.time, edge.label) for edge in random_walk
    ]
    # else:


def clean_random_walk(
    rw,
):  ### essentially if next time stamp is same then it make sures not to repeat the same node again ###
    newrw = [rw[0]]
    cur_time = rw[1][2]
    cur_nodes = [rw[1][0], rw[1][1]]
    newrw.append(rw[1])
    for wk in rw[2:]:

        if wk[2] == cur_time:
            if wk[1] in cur_nodes:
                return newrw
            else:
                newrw.append(wk)
                cur_nodes.append(wk[1])
        else:
            newrw.append(wk)
            cur_time = wk[2]
            cur_nodes = [wk[0], wk[1]]
    return newrw


def filter_rw(rw, cut_off=6):
    if len(rw) >= cut_off:
        return True
    else:
        return False


def convert_walk_to_seq(rw):
    seq = [(rw[1][0], rw[0])]
    for item in rw[1:]:
        seq.append((item[1], item[2]))
    return seq


def convert_seq_to_id(vocab, seq):
    nseq = []
    for item in seq:
        nseq.append((vocab[item[0]], item[1]))
    return nseq


def update_delta(delta, d=0.1):
    if delta == 0:
        return delta + d
    return delta


def get_time_delta(sequence, start_delta=0):
    times = [item[1] for item in sequence]
    delta = [update_delta(a - b) for a, b in zip(times[1:], times[:-1])]

    delta = [update_delta(0)] + delta  # + [-1]
    return [(item[0], item[1], t) for item, t in zip(sequence, delta)]

def get_node_set_length(edges):
    nodes = set()
    for start, end, _ in edges:
        nodes.add(start)
        nodes.add(end)
    return len(nodes)
