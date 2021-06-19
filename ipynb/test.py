import sys, os, platform, pickle, json, time
from datetime import datetime, timedelta
import numpy as np
from collections import Counter, defaultdict
import networkx as nx
from tqdm import tqdm
import math
import scipy as sp
import scipy.stats
from collections import defaultdict 

from utils.data_loader import DataLoader
from utils.plot import ColorPalette, concise_fmt, hide_spines, stackedBarPlot
data_prefix = 'data/' # ../

from powerlaw import Fit, plot_ccdf

import matplotlib as mpl
if platform.system() == 'Linux':
    mpl.use('Agg')  # no UI backend

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
mpl.rcParams['lines.linewidth'] = 1

if not os.path.exists(os.path.join(data_prefix, "test_strong_bridges.pkl")):
    test = nx.strongly_connected_components(graph_lst[0])
    lscc_test = max(test, key=len)
    test_subgraph = graph_lst[0].subgraph(lscc_test, )
    with open(os.path.join(data_prefix, "test_strong_bridges.pkl"), 'wb') as fout:
        pickle.dump(test_subgraph.copy(), fout)
else:
    with open(os.path.join(data_prefix, "test_strong_bridges.pkl"), 'rb') as fin:
        test_subgraph = pickle.load(fin)

def dfs_edges(v, graph):
    # parent = {}
    start = defaultdict(int)
#     visited = defaultdict(bool)
    end = defaultdict(int)
    # entering u
    forward_edges = defaultdict(list)
    back_edges = defaultdict(list)
    cross_edges = defaultdict(list)
    tree_edges = []
    t = 0
    print("entering dfs edges")

    def dfs(v, graph, parent=None):
        nonlocal t
        t += 1
        start[v] = t
        if parent is not None:
            tree_edges.append((parent, v))
        for n in graph.successors(v):
            if start[n] == 0:  # not visited
                dfs(n, graph, parent=v)
            elif end[n] == 0:
                back_edges[n].append(v)
            elif start[v] < start[n]:
                forward_edges[n].append(v)
            else:
                cross_edges[n].append(v)
        t += 1
        end[v] = t
    
    dfs(v, graph)
    
    return tree_edges, back_edges, forward_edges, cross_edges

def dominator_tree(v, graph):
    dt = nx.DiGraph()
    print("entering dominator_tree")
    idoms = nx.immediate_dominators(graph, v)
    for k, v in idoms.items():
        if k != v:
            dt.add_edge(v, k)
    return dt

def dfs_order(v, graph, ):
    start = defaultdict(int)
    end = defaultdict(int)
    t = 0
    print("entering dfs order")
    def dfs(v, graph,):
        nonlocal t
        t += 1
        start[v] = t
        for n in graph.successors(v):
            if start[n] == 0:  # not visited
                dfs(n, graph, )
        t += 1
        end[v] = t
        
    dfs(v, graph)
    
    return start, end

def edge_dominators(v, graph):    
    edge_dominator = set()
    tree_edges, back_edges, forward_edges, cross_edges = dfs_edges(v, graph)
    print("finish dfs edges")
    dt = dominator_tree(v, graph)
    print("finish dominator_tree")
    start, end = dfs_order(v, dt)
    print("finish dfs_order")
    for e in tree_edges:
        if len(forward_edges[e[1]])==0 and len(cross_edges[e[1]])==0:
            flag = True
            for i in back_edges[e[1]]:  # for every back edge, e_1 dominate i
                if not ((start[e[1]] < start[i]) and (end[i] < end[e[1]])):
                    flag = False
            if flag:
                edge_dominator.add(e)
    return edge_dominator

def strong_edges(v, graph):
    print("here")
    de = edge_dominators(v, graph)
    for v, u in edge_dominators(v, graph.reverse()):
        de.add((u, v))
    return de

de = edge_dominators(0, test_subgraph)

se = strong_edges(0, test_subgraph)